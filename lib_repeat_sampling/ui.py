from lib_repeat_sampling import global_state
from modules import script_callbacks, shared
from typing import Dict, Tuple, List, Callable
import gradio as gr
import dataclasses
from modules import sd_samplers

from modules.shared import opts
import modules.shared as shared


@dataclasses.dataclass
class AccordionInterface:
    get_elem_id: Callable

    def __post_init__(self):
        self.is_rendered = False

        self.checkbox = gr.Checkbox(label="Repeat enabled", value=False)
        self.repeat_denoise_strength = gr.Slider(label="Denoising strength", minimum=0.0, maximum=1.0, value=0.45, step=0.01)
        self.repeats = gr.Slider(1, 10, 1, label="Repeats", step=1, scale=3)
        self.fixed_seed = gr.Radio(label="Fix RNG", value="Independent", choices=["False", "Independent", "Same"], scale=1)
        self.tactic = gr.Radio(["Equal", "Decreasing"], label="Keep the denosing strength equal over several repeats, or decrease it", value="Equal")
        self.factor = gr.Slider(0, 1, 1, label="Factor decreasing denoising strength", step=0.01, visible=self.tactic.value == "Decreasing")
        self.sampler_name = gr.Dropdown(
            label="Sampling method", choices=[x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers], value="Use same sampler"
        )

    def arrange_components(self, is_img2img: bool):
        if self.is_rendered:
            return

        with gr.Accordion(label="Repeated Sampling", open=False):
            self.checkbox.render()
            with gr.Accordion(label="Advanced Settings", open=False):
                with gr.Row(variant="compact"):
                    self.sampler_name.render()
                    self.repeat_denoise_strength.render()
                with gr.Row(variant="compact"):
                    self.repeats.render()
                    self.fixed_seed.render()
                self.tactic.render()
                self.factor.render()

    def connect_events(self, is_img2img: bool):
        if self.is_rendered:
            return

        def set_tactic(tactic: str, factor: float):
            if tactic == "Equal":
                global_state.factor = 1
                return gr.update(visible=False)
            else:
                global_state.factor = factor
                return gr.update(visible=True)

        self.tactic.change(set_tactic, inputs=[self.tactic, self.factor], outputs=self.factor)

    def set_rendered(self, value: bool = True):
        self.is_rendered = value

    def get_components(self) -> Tuple[gr.components.Component]:
        return (self.checkbox, self.sampler_name, self.repeat_denoise_strength, self.repeats, self.fixed_seed, self.tactic, self.factor)

    def get_infotext_fields(self) -> Tuple[Tuple[gr.components.Component, str]]:
        return tuple(
            zip(
                self.get_components(),
                (
                    "Enable repeated sampling",
                    "Repeat Sample denoise_strength",
                    "How many times to repeat the sampling",
                    "Do denoising with same RNG-seed, increases detail, danger of burn in effect.",
                    (
                        "How the denoising strength should be calculated for multiple repeats"
                        "Each repeats denoising strength will be lowered by this factor compared to the step before"
                    ),
                ),
            )
        )

    def get_paste_field_names(self) -> List[str]:
        return [
            "Repeat enabled",
            "Repeat Sample sampler_name",
            "Repeat Sample denoise_strength",
            "Repeat Sample repeats",
            "Repeat Sample fix seed",
            "Repeat Sample repeat_tactic",
            "Repeat Sample factor",
        ]

    def get_extra_generation_params(self, args: Dict) -> Dict:
        return {
            "Repeat enabled": args["checkbox"],
            "Repeat Sample sampler_name": args["sampler_name"],
            "Repeat Sample denoise_strength": args["repeat_denoise_strength"],
            "Repeat Sample repeats": args["repeats"],
            "Repeat Sample fix seed": args["fixed_seed"],
            "Repeat Sample repeat_tactic": args["tactic"],
            "Repeat Sample factor": args["factor"],
        }

    def unpack_processing_args(
        self,
        checkbox: bool,
        sampler_name: str,
        repeat_denoise_strength: float,
        repeats: int,
        fixed_seed: bool,
        tactic: str,
        factor: float,
    ) -> Dict:
        return {
            "checkbox": checkbox,
            "sampler_name": sampler_name,
            "repeat_denoise_strength": repeat_denoise_strength,
            "repeats": repeats,
            "fixed_seed": fixed_seed,
            "tactic": tactic,
            "factor": factor,
        }


def on_ui_settings():
    section = ("repeat_denoise_strength", "Repeat Sampling")

    shared.opts.add_option("repeat_sampling_enabled", shared.OptionInfo(True, "Enable Repeat Sampling extension", section=section))
    global_state.is_enabled = shared.opts.data.get("repeat_sampling_enabled", True)

    shared.opts.add_option("repeat_sampling_verbose", shared.OptionInfo(False, "Enable verbose debugging for Repeat Sampling", section=section))
    shared.opts.onchange("repeat_sampling_verbose", update_verbose)


script_callbacks.on_ui_settings(on_ui_settings)


def update_verbose():
    global_state.verbose = shared.opts.data.get("repeat_sampling_verbose", False)


def on_after_component(component, **_kwargs):
    if getattr(component, "elem_id", None) == "txt2img_prompt":
        global txt2img_prompt_textbox
        txt2img_prompt_textbox = component

    if getattr(component, "elem_id", None) == "img2img_prompt":
        global img2img_prompt_textbox
        img2img_prompt_textbox = component


script_callbacks.on_after_component(on_after_component)
