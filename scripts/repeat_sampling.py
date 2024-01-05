from copy import copy
import random
import traceback
from lib_repeat_sampling import global_state, ui, xyz_grid
from modules import scripts, processing, shared
from typing import Dict
import inspect

import torch
from modules.shared import opts
from modules.processing import (
    StableDiffusionProcessing as Processing,
    StableDiffusionProcessingTxt2Img as ProcessingTxt2Img,
    StableDiffusionProcessingImg2Img as ProcessingImg2Img,
    create_infotext,
    process_images,
    get_fixed_seed,
)
import modules.scripts
import math
import PIL
import numpy as np

import modules.sd_hijack
from modules import (
    devices,
    sd_samplers,
    rng,
)
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
import modules.face_restoration
import modules.styles
import modules.sd_models as sd_models

from PIL import Image

import modules.sd_hijack
import modules.images as images
import modules.styles


class RepeatSamplingScript(scripts.Script):
    def __init__(self) -> None:
        self.accordion_interface = None
        self._is_img2img = False
        self.once = True
        self.sum_steps = None
        self.repeat_pass = False

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

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """

        pass

    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_image(self, p, pp: scripts.PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def setup(self, p, *args):
        try:
            args = self.accordion_interface.unpack_processing_args(*args)
            self.update_global_state(args)
        except BaseException:
            return
        if not global_state.checkbox:
            return
        if global_state.checkbox:
            # state.job_count *= global_state.repeats
            if not shared.state.processing_has_refined_job_count:
                if shared.state.job_count == -1:
                    shared.state.job_count = p.n_iter

                shared.total_tqdm.updateTotal((p.steps + (p.steps) * global_state.repeats) * shared.state.job_count)
                shared.state.job_count = shared.state.job_count * (1 + global_state.repeats)
                shared.state.processing_has_refined_job_count = True

    def before_process_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        if not global_state.checkbox:
            return
        global_state.factor = 1 if global_state.tactic == "Equal" else global_state.factor
        if global_state.checkbox and self.repeat_pass == False:
            self.accordion_interface.apply_and_clear_repeat_override()
            global_state.repeats_reset = global_state.repeats
            p.start_steps = p.steps
            p.sum_steps = p.steps
            act_steps = shifted_steps = last_steps = 0
            while last_steps < global_state.min_step or act_steps <= 3:
                first_steps = self.geometric_series(p.start_steps)
                local_repeat_denoise_strength = global_state.repeat_denoise_strength
                noise_schedule = [
                    local_repeat_denoise_strength * global_state.factor ** (x) for x in range(global_state.repeats)
                ]
                step_schedule = [int(first_steps * y) for y in noise_schedule]
                normalized_steps = [int(step / noise) for step, noise in zip(step_schedule, noise_schedule)]
                last_steps = normalized_steps[-1]
                act_steps = step_schedule[-1]
                if last_steps < global_state.min_step or act_steps <= 3:
                    shifted_steps = p.start_steps - p.steps

                    if (
                        step_schedule[0] <= self.geometric_series(total_steps=p.start_steps) - (shifted_steps + 1)
                        and global_state.min_step
                        <= self.geometric_series(total_steps=p.start_steps) - shifted_steps - 1
                    ):
                        p.start_steps += 1
                    else:
                        global_state.repeats -= 1
                        # shared.state.job_count -= 1
                        p.start_steps = p.steps
                        print(
                            f"Last repeat would be only {last_steps} steps, reducing repeats to {global_state.repeats}"
                        )

            if shifted_steps > 0:
                print(f"Shifted {shifted_steps} steps into the repeats")
            p.steps = self.geometric_series(total_steps=p.start_steps) - shifted_steps
            p.start_steps = shifted_steps + p.steps

            p.denoising_strength = global_state.repeat_denoise_strength
            print(noise_schedule)
            print(step_schedule)
            print(normalized_steps)

            args = self.accordion_interface.unpack_processing_args(*args)
            p.extra_generation_params.update(self.accordion_interface.get_extra_generation_params(args))

    def geometric_series(self, total_steps):
        return math.ceil(
            (
                total_steps
                / (
                    1
                    + global_state.repeat_denoise_strength
                    * (1 - global_state.factor**global_state.repeats)
                    / (1 - global_state.factor)
                )
                if global_state.factor != 1
                else total_steps / (1 + global_state.repeat_denoise_strength * global_state.repeats)
            )
        )

    def postprocess_batch_list(
        self,
        p_o: processing.StableDiffusionProcessing,
        pp: modules.scripts.PostprocessBatchListArgs,
        *script_args,
        batch_number,
        **kwargs,
    ):
        if not global_state.checkbox:
            return
        # global_state.self.accordion_interface.apply_and_clear_repeat_override()
        if global_state.checkbox and self.repeat_pass == False:

            def to_tensor(image):
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                return torch.Tensor(image)

            self_sampling = True
            self.repeat_pass = True
            self.accordion_interface.apply_and_clear_repeat_override()
            p: processing.StableDiffusionProcessing = copy(p_o)
            p.n_iter = 1
            try:
                p.seed = p.all_seeds
                p.prompt = p.all_prompts
                p.negative_prompt = p.all_negative_prompts
                p.subseed = p.all_subseeds
            except BaseException:
                p.seed = get_fixed_seed(p.seed)
            local_repeat_denoise_strength = global_state.repeat_denoise_strength
            try:
                with SanityCount(expectation=p.sum_steps, init=p.steps, init_seed=p.seed, original_p=p_o) as sc:
                    p.steps = p.start_steps

                    img2img_sampler_name = (
                        p.sampler_name if global_state.sampler_name == "Use same sampler" else global_state.sampler_name
                    )
                    decoded_samples = pp.images
                    if global_state.fixed_seed == global_state.FO_INDEPENDENT:
                        # p.seed = int(random.randrange(4294967294))
                        p.all_seeds = [int(random.randrange(4294967294)) for seed in p.all_seeds]
                    for i in range(global_state.repeats):
                        if (
                            global_state.fixed_seed == global_state.FO_FALSE
                            and not global_state.fixed_seed_extra == global_state.FOE_OTHER
                        ):
                            p.all_seeds = [seed + 1 for seed in p.all_seeds]
                        if global_state.fixed_seed_extra == global_state.FOE_LAST and i == global_state.repeats - 1:
                            p.all_seeds = [int(random.randrange(4294967294)) for seed in p.all_seeds]
                        # Define a dictionary of actions for each fixed seed
                        # value
                        actions = {
                            global_state.FO_INDEPENDENT: lambda seed, i: seed + 1 if i % 2 == 0 else seed - 1,
                            global_state.FO_SAME: lambda seed, i: seed + 1 if i % 2 == 1 else seed - 1,
                            global_state.FO_FALSE: lambda seed, i: seed + 1 if i % 2 == 0 else seed,
                        }

                        # Check if the fixed seed extra is alternate
                        if global_state.fixed_seed_extra == global_state.FOE_ALTERNATE:
                            # Get the action for the current fixed seed value
                            action = actions.get(global_state.fixed_seed)
                            # If the action is defined, update the seed
                            # accordingly
                            if action:
                                p.all_seeds = [action(seed, i) for seed in p.all_seeds]
                        p.seed = p.all_seeds
                        p.prompt = p.all_prompts
                        p.negative_prompt = p.all_negative_prompts
                        p.subseed = p.all_subseeds
                        opt_C = 4
                        opt_f = 8
                        p.rng = rng.ImageRNG((opt_C, p.height // opt_f, p.width // opt_f), p.seed, subseeds=p.subseed, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)

                        if self_sampling:
                            image = torch.stack(decoded_samples)
                            image = image.to(shared.device, dtype=devices.dtype_vae)
                            init_latent = images_tensor_to_samples(
                                image, approximation_indexes.get(opts.sd_vae_encode_method), p.sd_model
                            )
                            p.denoising_strength = local_repeat_denoise_strength
                            p.scripts = None
                            samples = self.sample_pass(p, init_latent, p.seed, p.prompt, img2img_sampler_name)
                            shared.state.job_count += 1
                            shared.state.nextjob()
                            sc(int(p.steps * p.denoising_strength), p.denoising_strength, p.seed)
                            decoded_samples = processing.decode_latent_batch(
                                p.sd_model, samples, target_device=devices.cpu, check_for_nans=True
                            )
                            decoded_samples = [
                                torch.clamp((x.float() + 1.0) / 2.0, min=0.0, max=1.0) for x in decoded_samples
                            ]
                        else:
                            img = [
                                PIL.Image.fromarray((255.0 * np.moveaxis(x.cpu().numpy(), 0, 2)).astype(np.uint8))
                                for x in decoded_samples
                            ]

                            pc = update_img2img_p(p, img, global_state.repeat_denoise_strength)
                            # self.old_version(p, pp, sc, img2img_sampler_name)
                            # self.process(pc, *script_args, **kwargs)
                            pc.denoising_strength = local_repeat_denoise_strength
                            # pc.seed = p.seed
                            pc.sampler_name = img2img_sampler_name
                            pc.do_not_save_samples = True
                            try:
                                pc.enable_hr = p_o.pc.enable_hr
                            except AttributeError:
                                pc.enable_hr = 0
                            pc.scripts = None
                            pc = process_images(pc)
                            shared.state.job_count += 1
                            shared.state.nextjob()
                            sc(int(pc.steps * pc.denoising_strength), pc.denoising_strength, pc.seed)
                            decoded_samples = [to_tensor(x) for x in pc.images]
                        local_repeat_denoise_strength *= global_state.factor

            except Exception as e:
                traceback.print_exc()
                print(e)
            if global_state.return_only_result:
                pp.images = [x for x in decoded_samples[-p.batch_size :]]
            else:
                pp.images.extend([x for x in decoded_samples])
                p.seeds = p.seeds + p.seeds[-p.batch_size :] * p.batch_size
                p.prompts = p.prompts + p.prompts[-p.batch_size :] * p.batch_size
                p.negative_prompts = p.negative_prompts + p.negative_prompts[-p.batch_size :] * p.batch_size

            self.repeat_pass = False
            p_o.prompts = p.prompts
            p_o.negative_prompts = p.negative_prompts
            p_o.seeds = p.seeds
            p_o.steps = p.sum_steps
            if global_state.repeats_reset != global_state.repeats:
                global_state.repeats = global_state.repeats_reset

    def sample_pass(_, self, samples, seeds, prompts, img2img_sampler_name):
        if shared.state.interrupted:
            return samples

        self.is_hr_pass = True

        def save_intermediate(image, index):
            """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

            if global_state.return_only_result:
                return

            if not isinstance(image, Image.Image):
                image = sd_samplers.sample_to_image(image, index, approximation=0)

            info = create_infotext(
                self,
                self.all_prompts,
                self.all_seeds,
                self.all_subseeds,
                [],
                iteration=self.iteration,
                position_in_batch=index,
            )
            images.save_image(
                image,
                self.outpath_samples,
                "",
                seeds[index],
                prompts[index],
                opts.samples_format,
                info=info,
                p=self,
                suffix="-before-highres-fix",
            )

        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)

        for i in range(samples.shape[0]):
            save_intermediate(samples, i)

        noise = self.rng.next()

        # GC now before running the next img2img to prevent running out of memory
        devices.torch_gc()

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())

        # if self.scripts is not None:
        #     self.scripts.before_hr(self)

        samples = self.sampler.sample_img2img(
            self,
            samples,
            noise,
            self.c,
            self.uc,
            steps=None,
            image_conditioning=samples.new_zeros(samples.shape[0], 5, 1, 1),
        )

        self.sampler = None
        devices.torch_gc()
        self.is_hr_pass = False
        return samples

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

        self.accordion_interface.apply_and_clear_repeat_override()


def apply_and_clear_repeat_override():
    locals_here = list(vars(global_state).keys())
    for x in locals_here:
        try:
            override = vars(global_state).get(f"{x}_override")
            if override is not None:
                vars(global_state)[x] = override
                vars(global_state)[f"{x}_override"] = None
        except BaseException:
            pass


xyz_grid.patch()


class SanityCount:
    def __init__(self, expectation, original_p: Processing, init=0, init_seed=-1):
        self.count = init
        self.expectation = expectation
        self.noise = list()
        self.noise_t = list()
        self.init_seed = init_seed
        self.og = original_p
        print(f"Entered repeat sampling loop with the options:")
        print(f"'global_state.fixed_seed': {global_state.fixed_seed},")
        print(f"'global_state.fixed_seed_extra': {global_state.fixed_seed_extra},")
        print(f"'global_state.min_step': {global_state.min_step},")
        print(f"'global_state.sampler_name': {global_state.sampler_name},")
        print(f"'global_state.repeat_denoise_strength': {global_state.repeat_denoise_strength},")
        print(f"'global_state.tactic': {global_state.tactic},")
        print(f"'global_state.repeats': {global_state.repeats},")
        print(f"'global_state.repeats_reset': {global_state.repeats_reset},")
        print(f"'global_state.return_only_result': {global_state.return_only_result},")

    def __enter__(self):
        return self

    def __call__(self, val, noise=None, n_t=None):
        self.count += val
        if noise is not None:
            self.noise.append(noise)
        if n_t is not None:
            self.noise_t.append(n_t)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print()
        print(f"Steps done: {self.count}/{self.expectation}")
        print("noise: ", self.noise)
        print("start seed:", self.init_seed, ", seed: ", self.noise_t)
        self.og.seeds = [self.init_seed] + self.noise_t
        print()


def update_img2img_p(p: Processing, imgs: PIL.Image.Image, denoising_strength: float = 0.75) -> ProcessingImg2Img:
    if isinstance(p, ProcessingImg2Img):
        p.init_images = imgs
        p.denoising_strength = denoising_strength
        return p

    if isinstance(p, ProcessingTxt2Img):
        kwargs = {
            k: getattr(p, k)
            for k in dir(p)
            if k
            in list(inspect.signature(modules.processing.StableDiffusionProcessing).parameters)
            + list(inspect.signature(ProcessingImg2Img).parameters)
        }  # inherit params
        kwargs["denoising_strength"] = denoising_strength
        pc = ProcessingImg2Img(
            init_images=imgs,
            **kwargs,
        )
        pc.scripts = modules.scripts.scripts_txt2img
        pc.script_args = p.script_args
        return pc
