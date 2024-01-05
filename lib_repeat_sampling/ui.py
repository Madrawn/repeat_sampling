from lib_repeat_sampling import global_state
from modules import script_callbacks, shared
from typing import Dict, Tuple, List, Callable
import gradio as gr
import dataclasses
from modules import sd_samplers

from modules.shared import opts
import modules.shared as shared
from modules.ui_components import InputAccordion


@dataclasses.dataclass
class AccordionInterface:
    get_elem_id: Callable

    def __post_init__(self):
        self.is_rendered = False

        self.repeat_denoise_strength = gr.Slider(
            label="Denoising strength", minimum=0.0, maximum=1.0, value=0.45,
            step=0.01, elem_id=make_element_id("repeat_denoise_strength"))
        self.return_only_result = gr.Checkbox(
            value=True, label="Return only result image",
            elem_id=make_element_id("return_only_result"))
        self.repeats = gr.Slider(1, 10, 1, label="Repeats", step=1,
                                 elem_id=make_element_id("repeats"))
        self.min_step = gr.Slider(
            minimum=1,
            maximum=100,
            value=15,
            label="Minimum Steps",
            step=1,
            elem_id=make_element_id("min_step"))
        self.fixed_seed = gr.Dropdown(
            label="Fix RNG", value="Independent",
            choices=global_state.fixed_seed_options,
            elem_id=make_element_id("fixed_seed"))
        self.fixed_seed_extra = gr.Dropdown(
            label="Fix RNG Extra", value="Normal",
            choices=global_state.fixed_seed_extra_options,
            elem_id=make_element_id("fixed_seed_extra"))
        self.tactic = gr.Radio(
            ["Equal", "Decreasing"],
            label="Keep the denosing strength equal over several repeats, or decrease it",
            value="Equal", elem_id=make_element_id("tactic"))
        self.factor = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.75,
            label="Factor decreasing denoising strength",
            step=0.01,
            visible=self.tactic.value == "Decreasing",
            elem_id=make_element_id("factor")

        )
        self.sampler_name = gr.Dropdown(
            label="Sampling method", choices=["Use same sampler"] +
            [x.name for x in sd_samplers.samplers
             if x.name not in opts.hide_samplers],
            value="Use same sampler", elem_id=make_element_id("sampler_name"))

    def arrange_components(self, is_img2img: bool):
        if self.is_rendered:
            return
        with gr.Blocks():
            with InputAccordion(label="Repeated Sampling", value=False, elem_id=make_element_id("InputAccordion")) as self.checkbox:
                # with gr.Accordion(label="Repeated Sampling", open=False):
                with gr.Accordion(label="Advanced Settings", open=False, elem_id=make_element_id("Accordion")):
                    with gr.Row(variant="compact"):
                        self.sampler_name.render()
                        self.repeat_denoise_strength.render()
                    with gr.Row(variant="compact", elem_id=make_element_id("Row")):
                        with gr.Column(elem_id=make_element_id("Column1")):
                            self.return_only_result.render()
                            self.repeats.render()
                        with gr.Column(elem_id=make_element_id("Column2")):
                            self.fixed_seed.render()
                            self.fixed_seed_extra.render()
                    self.tactic.render()
                    self.factor.render()
                    self.min_step.render()

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

        # def set_seed_extra(extra: str):
        #     if extra == "False":
        #         return gr.update(visible=False)
        #     else:
        #         return gr.update(visible=True)

        self.tactic.change(
            set_tactic,
            inputs=[
                self.tactic,
                self.factor],
            outputs=self.factor)
        # self.fixed_seed.change(set_seed_extra, inputs=[self.fixed_seed], outputs=self.fixed_seed_extra)

    def set_rendered(self, value: bool = True):
        self.is_rendered = value

    def get_components(self) -> Tuple[gr.components.Component]:
        self.update_global_state(
            {
                "checkbox": self.checkbox.value,
                "sampler_name": self.sampler_name.value,
                "repeat_denoise_strength": self.repeat_denoise_strength.value,
                "return_only_result": self.return_only_result.value,
                "repeats": self.repeats.value,
                "fixed_seed": self.fixed_seed.value,
                "fixed_seed_extra": self.fixed_seed_extra.value,
                "tactic": self.tactic.value,
                "factor": self.factor.value,
                "min_step": self.min_step.value,
            }
        )
        return (
            self.checkbox,
            self.sampler_name,
            self.repeat_denoise_strength,
            self.return_only_result,
            self.repeats,
            self.fixed_seed,
            self.fixed_seed_extra,
            self.tactic,
            self.factor,
            self.min_step,
        )

    def get_infotext_fields(
            self) -> Tuple[Tuple[gr.components.Component, str]]:
        return tuple(
            zip(
                self.get_components(),
                ("Enable repeated sampling",
                 "Repeat Sample denoise_strength",
                 "How many times to repeat the sampling",
                 "Do denoising with same RNG-seed, increases detail, danger of burn in effect.",
                 ("How the denoising strength should be calculated for multiple repeats"
                  "Each repeats denoising strength will be lowered by this factor compared to the step before"),
                 ),
            ))

    def get_paste_field_names(self) -> List[str]:
        return [
            "Repeat enabled",
            "Repeat Sample sampler_name",
            "Repeat Sample denoise_strength",
            "Repeat Sample return_only_result",
            "Repeat Sample repeats",
            "Repeat Sample fix seed",
            "Repeat Sample fix seed extra",
            "Repeat Sample repeat_tactic",
            "Repeat Sample factor",
            "Repeat Sample min_step",
        ]

    def get_extra_generation_params(self, args: Dict) -> Dict:
        return {
            "Repeat enabled": args["checkbox"],
            "Repeat Sample sampler_name": args["sampler_name"],
            "Repeat Sample denoise_strength": args["repeat_denoise_strength"],
            "Repeat Sample return_only_result": args["return_only_result"],
            "Repeat Sample repeats": args["repeats"],
            "Repeat Sample fix seed": args["fixed_seed"],
            "Repeat Sample fix seed extra": args["fixed_seed_extra"],
            "Repeat Sample repeat_tactic": args["tactic"],
            "Repeat Sample factor": args["factor"],
            "Repeat Sample min_step": args["min_step"],
        }

    def unpack_processing_args(
        self,
        checkbox: bool,
        sampler_name: str,
        repeat_denoise_strength: float,
        return_only_result: bool,
        repeats: int,
        fixed_seed: str,
        fixed_seed_extra: str,
        tactic: str,
        factor: float,
        min_step: int,
    ) -> Dict:
        return {
            "checkbox": checkbox,
            "sampler_name": sampler_name,
            "repeat_denoise_strength": repeat_denoise_strength,
            "return_only_result": return_only_result,
            "repeats": repeats,
            "fixed_seed": fixed_seed,
            "fixed_seed_extra": fixed_seed_extra,
            "tactic": tactic,
            "factor": factor,
            "min_step": min_step,
        }

    def update_global_state(self, args: Dict):
        if shared.state.job_no == 0:
            global_state.is_enabled = shared.opts.data.get(
                "repeat_sampling_enabled", True)

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

        self.apply_and_clear_repeat_override()

    def apply_and_clear_repeat_override(self):
        locals_here = list(vars(global_state).keys())
        for x in locals_here:
            try:
                override = vars(global_state).get(f"{x}_override")
                if override is not None:
                    vars(global_state)[x] = override
                    vars(global_state)[f"{x}_override"] = None
            except BaseException:
                pass


def on_ui_settings():
    section = ("repeat_denoise_strength", "Repeat Sampling")

    shared.opts.add_option(
        "repeat_sampling_enabled",
        shared.OptionInfo(
            True,
            "Enable Repeat Sampling extension",
            section=section))
    global_state.is_enabled = shared.opts.data.get(
        "repeat_sampling_enabled", True)

    shared.opts.add_option("repeat_sampling_verbose", shared.OptionInfo(
        False, "Enable verbose debugging for Repeat Sampling", section=section), )
    shared.opts.onchange("repeat_sampling_verbose", update_verbose)


script_callbacks.on_ui_settings(on_ui_settings)


def update_verbose():
    global_state.verbose = shared.opts.data.get(
        "repeat_sampling_verbose", False)


def on_after_component(component, **_kwargs):
    if getattr(component, "elem_id", None) == "txt2img_prompt":
        global txt2img_prompt_textbox
        txt2img_prompt_textbox = component

    if getattr(component, "elem_id", None) == "img2img_prompt":
        global img2img_prompt_textbox
        img2img_prompt_textbox = component


def make_element_id(name: str) -> str:
    return "rpts-" + name


script_callbacks.on_after_component(on_after_component)
