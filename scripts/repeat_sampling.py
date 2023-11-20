from lib_repeat_sampling import global_state, ui, xyz_grid
from modules import scripts, processing, shared
from typing import Dict
from contextlib import closing

import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr
import json
import logging
import math
import os
import sys
import hashlib
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image, ImageOps
import random
import cv2
from skimage import exposure
from typing import Any

import modules.sd_hijack
from modules import (
    devices,
    prompt_parser,
    masking,
    sd_samplers,
    lowvram,
    generation_parameters_copypaste,
    extra_networks,
    sd_vae_approx,
    scripts,
    sd_samplers_common,
    sd_unet,
    errors,
    rng,
    sd_clip,
)
from modules.rng import slerp  # noqa: F401
from modules.sd_hijack import model_hijack
from modules.sd_samplers_common import images_tensor_to_samples, decode_first_stage, approximation_indexes, setup_img2img_steps
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion

from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType


class RepeatSamplingScript(scripts.Script):
    def __init__(self):
        self.accordion_interface = None
        self._is_img2img = False

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

    def process(self, p: processing.StableDiffusionProcessing, *args):
        args = self.accordion_interface.unpack_processing_args(*args)
        self.update_global_state(args)
        global_state.factor = 1 if global_state.tactic == "Equals" else global_state.factor
        if global_state.checkbox:
            p.steps = math.ceil(
                p.steps / (1 + global_state.repeat_denoise_strength * (1 - global_state.factor**global_state.repeats) / (1 - global_state.factor))
            )
            p.denoising_strength = args["repeat_denoise_strength"]
            p.extra_generation_params.update(self.accordion_interface.get_extra_generation_params(args))

    def postprocess_batch_list(self, p: processing.StableDiffusionProcessing, pp: modules.scripts.PostprocessBatchListArgs, *script_args, **kwargs):
        if global_state.checkbox:
            img2img_sampler_name = p.sampler_name

            p.sampler = sd_samplers.create_sampler(img2img_sampler_name, p.sd_model)

            # GC now before running the next img2img to prevent running out of memory
            devices.torch_gc()

            with devices.autocast():
                extra_networks.activate(p, p.extra_network_data)
            sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
            images = torch.stack(pp.images).to(device=shared.device, dtype=devices.dtype_vae)
            samples = images_tensor_to_samples(images, approximation_indexes.get(opts.sd_vae_encode_method))
            p.rng = rng.ImageRNG(samples.shape[1:], p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength)
            p.rng.next()
            noise = p.rng.next()

            image_conditioning = p.txt2img_image_conditioning(images)
            sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
            samples_ddim = p.sampler.sample_img2img(p, samples, noise, p.c, p.uc, steps=None, image_conditioning=image_conditioning)
            x_samples_ddim = processing.decode_latent_batch(p.sd_model, samples_ddim, target_device=devices.cpu, check_for_nans=True)
            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            pp.images = x_samples_ddim

            p.sampler = None
            devices.torch_gc()

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


xyz_grid.patch()
