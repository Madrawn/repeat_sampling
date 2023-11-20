import sys
from types import ModuleType
from typing import Optional, Type
from modules import scripts
from lib_repeat_sampling import global_state


def patch():
    xyz_module = find_xyz_module()
    if xyz_module is None:
        print("[repeat sampling]", "xyz_grid.py not found.", file=sys.stderr)
        return

    xyz_module.axis_options.extend(
        [
            xyz_module.AxisOption("[repeat sampling] Repeat enabled", lambda x: bool(x), apply_repeat_field("checkbox", bool)),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample denoise_strength", int_or_float, apply_repeat_field("repeat_denoise_strength", float)),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample repeats", int_or_float, apply_repeat_field("repeats", int)),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample repeat_tactic", str, apply_repeat_field("tactic", str)),
            xyz_module.AxisOption("[repeat sampling] Repeat Sample factor", int_or_float, apply_repeat_field("factor", float)),
        ]
    )


class XyzFloat(float):
    is_xyz: bool = True


def apply_repeat_field(field: str, T: Type):
    def callback(_p, v, _vs):
        try:
            setattr(global_state, field, T(v))
        except:
            pass

    return callback


def int_or_float(string):
    try:
        return int(string)
    except ValueError:
        return float(string)


def find_xyz_module() -> Optional[ModuleType]:
    for data in scripts.scripts_data:
        if data.script_class.__module__ in {"xyz_grid.py", "xy_grid.py"} and hasattr(data, "module"):
            return data.module

    return None
