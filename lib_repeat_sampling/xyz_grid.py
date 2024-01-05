import sys
from types import ModuleType
from typing import Optional, Type
from modules import scripts
from lib_repeat_sampling import global_state
from modules import sd_samplers
from modules.shared import opts


def patch():
    xyz_module = find_xyz_module()
    if xyz_module is None:
        print("[repeat sampling]", "xyz_grid.py not found.", file=sys.stderr)
        return

    xyz_module.axis_options.extend(
        [
            xyz_module.AxisOption(
                "[repeat sampling] Repeat enabled",
                str,
                apply_repeat_field("checkbox", bool, translator=lambda bool: bool == "True"),
                choices=xyz_module.boolean_choice(),
            ),
            xyz_module.AxisOption(
                "[repeat sampling] Fixed seed",
                str,
                apply_repeat_field("fixed_seed", str),
                choices=lambda: global_state.fixed_seed_options,
            ),
            xyz_module.AxisOption(
                "[repeat sampling] Fixed seed extra",
                str,
                apply_repeat_field("fixed_seed_extra", str),
                choices=lambda: global_state.fixed_seed_extra_options,
            ),
            xyz_module.AxisOption(
                "[repeat sampling] sampler",
                str,
                apply_repeat_field("sampler_name", str),
                choices=lambda: [x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers],
            ),
            xyz_module.AxisOption(
                "[repeat sampling] Repeat Sample denoise_strength",
                float,
                apply_repeat_field("repeat_denoise_strength", float),
            ),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample repeats", int, apply_repeat_field("repeats", int)),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample min steps", int, apply_repeat_field("min_step", int)),
            xyz_module.AxisOption(
                "[repeat sampling] Repeat Sample repeat_tactic", str, apply_repeat_field("tactic", str)
            ),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample factor", float, apply_repeat_field("factor", float)),
        ]
    )


class XyzFloat(float):
    is_xyz: bool = True


def apply_repeat_field(field: str, T: Type, translator=lambda x: x):
    def callback(_p, v, _vs):
        try:
            setattr(global_state, f"{field}_override", T(translator(v)))
        except:
            pass

    return callback


# def int_or_float(string):
#     try:
#         return int(string)
#     except ValueError:
#         return float(string)


def find_xyz_module() -> Optional[ModuleType]:
    for data in scripts.scripts_data:
        if data.script_class.__module__ in {"xyz_grid.py", "xy_grid.py"} and hasattr(data, "module"):
            return data.module

    return None
