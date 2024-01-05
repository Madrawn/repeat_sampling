from typing import List, Optional


verbose: bool = True
is_enabled: bool = False
is_enabled_override: Optional[bool] = None
return_only_result: bool = False
repeat_denoise_strength: float = 0.45
repeat_denoise_strength_override: Optional[float] = None
checkbox: bool = False
checkbox_override: Optional[bool] = None
repeats_reset: int = 1
repeats: int = 1
repeats_override: Optional[int] = None
FO_FALSE = "False"
FO_INDEPENDENT = "Independent"
FO_SAME = "Same"
fixed_seed_options: list[str] = [FO_FALSE, FO_INDEPENDENT, FO_SAME]
fixed_seed: str = "False"
fixed_seed_override: Optional[str] = None
FOE_NORMAL = "Normal"
FOE_OTHER = FOE_ALTERNATE = "Every other"
FOE_LAST = "Last different"
fixed_seed_extra_options: list[str] = [FOE_NORMAL, FOE_OTHER, FOE_LAST]
fixed_seed_extra: str = "Normal"
fixed_seed_extra_override: Optional[str] = None
# fixed_seed: bool = False
# fixed_seed_override: Optional[bool] = None
tactic: str = "Equal"
tactic_override: Optional[str] = None
factor: float = 1
factor_override: Optional[float] = None
min_step: int = 3
min_step_override: Optional[int] = None
sampler_name: str = "Use same sampler"
sampler_name_override: Optional[str] = None
