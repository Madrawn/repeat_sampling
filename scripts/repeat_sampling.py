from lib_repeat_sampling import global_state, ui, xyz_grid
from modules import scripts, processing, shared
from typing import Dict

import modules.scripts
from modules.shared import opts, state
import math

import torch

import modules.sd_hijack
from modules import (
    devices,
    sd_samplers,
    extra_networks,
    rng,
)
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes, setup_img2img_steps
import modules.face_restoration
import modules.styles
import modules.sd_models as sd_models


class RepeatSamplingScript(scripts.Script):
    def __init__(self) -> None:
        self.accordion_interface = None
        self._is_img2img = False
        self.once = True
        self.sum_step = None

    @property
    def is_img2img(self):
        return self._is_img2img

    @is_img2img.setter
    def is_img2img(self, is_img2img):
        self._is_img2img = is_img2img
        if self.accordion_interface is None:
            self.accordion_interface = ui.AccordionInterface(self.elem_id)

    def title(self) -> str:
        return "Repeat Sampling"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        self.accordion_interface.arrange_components(is_img2img)
        self.accordion_interface.connect_events(is_img2img)
        self.infotext_fields = self.accordion_interface.get_infotext_fields()
        self.paste_field_names = self.accordion_interface.get_paste_field_names()
        self.accordion_interface.set_rendered()
        return self.accordion_interface.get_components()

    def setup(self, p, *args):
        if global_state.checkbox:
            self.sum_step = p.steps
            state.job_count += p.batch_size * global_state.repeats
            self.once = False

    def process(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        args = self.accordion_interface.unpack_processing_args(*args)
        self.update_global_state(args)
        if self.sum_step is not None:
            p.steps = self.sum_step
        global_state.factor = 1 if global_state.tactic == "Equal" else global_state.factor
        if global_state.checkbox:
            p.steps = math.ceil(
                (
                    (p.steps)
                    / (
                        1
                        + global_state.repeat_denoise_strength
                        * (1 - global_state.factor**global_state.repeats)
                        / (1 - global_state.factor)
                    )
                    if global_state.factor != 1
                    else p.steps / (1 + global_state.repeat_denoise_strength * global_state.repeats)
                )
            )
            self.start_step = p.steps
            p.denoising_strength = global_state.repeat_denoise_strength
            p.extra_generation_params.update(self.accordion_interface.get_extra_generation_params(args))

    def postprocess_batch_list(
        self,
        p: processing.StableDiffusionProcessing,
        pp: modules.scripts.PostprocessBatchListArgs,
        *script_args,
        **kwargs,
    ):
        # global_state.apply_and_clear_repeat_override()
        if global_state.checkbox:

            with SanityCount(self.sum_step, init=self.start_step) as sc:
                img2img_sampler_name = (
                    p.sampler_name if global_state.sampler_name == "Use same sampler" else global_state.sampler_name
                )
                noise = None
                
                for i in range(global_state.repeats):
                    if state.interrupted or state.skipped:
                        break
                    # GC now before running the next img2img to prevent running out of memory
                    p.sampler = sd_samplers.create_sampler(img2img_sampler_name, p.sd_model)
                    devices.torch_gc()

                    with devices.autocast():
                        extra_networks.activate(p, p.extra_network_data)
                    sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
                    images = torch.stack(pp.images).to(device=shared.device, dtype=devices.dtype_vae)
                    samples = images_tensor_to_samples(images, approximation_indexes.get(opts.sd_vae_encode_method))
                    if global_state.fixed_seed != "False":
                        p.rng = rng.ImageRNG(
                            samples.shape[1:], p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength
                        )
                        same = p.rng.next()
                        different = p.rng.next()

                    if global_state.fixed_seed == "Same":
                        noise = same

                    if global_state.fixed_seed == "Independent":
                        noise = different

                    if global_state.fixed_seed_extra == "Last different" and i == global_state.repeats - 1:
                        noise = p.rng.next()

                    if global_state.fixed_seed_extra == "Every other" and i % 2 == (
                        0 if global_state.fixed_seed != "Same" else 1
                    ):
                        noise = p.rng.next()

                    if global_state.fixed_seed == "False" and not global_state.fixed_seed_extra == "Every other":
                        noise = p.rng.next()

                    if noise is None:
                        noise = p.rng.next()
                        
                    image_conditioning = p.txt2img_image_conditioning(images)
                    sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
                    # _, x = setup_img2img_steps(p)
                    steps_to_take = round(max(p.steps * p.denoising_strength, global_state.min_step))
                    sc(steps_to_take, p.denoising_strength, noise[0,0,0,0].item())
                    samples_ddim = p.sampler.sample_img2img(
                        p, samples, noise, p.c, p.uc, steps=steps_to_take, image_conditioning=image_conditioning
                    )
                    if global_state.tactic != "Equal":
                        # p.steps = steps_to_take
                        p.denoising_strength *= global_state.factor

                    x_samples_ddim = processing.decode_latent_batch(
                        p.sd_model, samples_ddim, target_device=devices.cpu, check_for_nans=True
                    )
                    x_samples_ddim = torch.stack(x_samples_ddim).float()
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    pp.images = list(x_samples_ddim)
                    state.nextjob()

                p.sampler = None
                devices.torch_gc()
                p.steps = self.start_step  # reset steps to service potential batch
                p.denoising_strength = global_state.repeat_denoise_strength  # reset noise to service potential batch

    def update_global_state(self, args: Dict):
        if shared.state.job_no == 0:
            global_state.is_enabled = shared.opts.data.get("repeat_sampling_enabled", True)

        for k, v in args.items():
            try:
                getattr(global_state, k)
            except AttributeError:
                continue

            if getattr(getattr(global_state, k), "is_xyz", False):
                xyz_attr = getattr(global_state, k)
                xyz_attr.is_xyz = False
                args[k] = xyz_attr
                continue

            if shared.state.job_no > 0:
                continue

            setattr(global_state, k, v)

        apply_and_clear_repeat_override()


xyz_grid.patch()


def apply_and_clear_repeat_override():
    locals_here = list(vars(global_state).keys())
    for x in locals_here:
        try:
            override = vars(global_state).get(f"{x}_override")
            if override is not None:
                vars(global_state)[x] = override
                vars(global_state)[f"{x}_override"] = None
        except:
            pass


class SanityCount:
    def __init__(
        self,
        expectation,
        init=0,
    ):
        self.count = init
        self.expectation = expectation
        self.noise = list()
        self.noise_t = list()

    def __enter__(self):
        return self

    def __call__(self, val, noise, n_t):
        self.count += val
        if noise is not None:
            self.noise.append(noise)
        if n_t is not None:
            self.noise_t.append(n_t)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print()
        print(f"Steps done: {self.count}/{self.expectation}")
        print("noise: ", self.noise)
        print("noise tensor: ", self.noise_t)
        print()
