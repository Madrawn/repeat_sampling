from typing import List, Optional


verbose: bool = True
is_enabled: bool = False
is_enabled_override: Optional[bool] = None
repeat_denoise_strength: float = 0.45
repeat_denoise_strength_override: Optional[float] = None
checkbox: bool = False
checkbox_override: Optional[bool] = None
repeats: int = 1
repeats_override: Optional[int] = None
fixed_seed: str = "False"
fixed_seed_override: Optional[str] = None
# fixed_seed: bool = False
# fixed_seed_override: Optional[bool] = None
tactic: str = "Equal"
tactic_override: Optional[str] = None
factor: float = 1
factor_override: Optional[float] = None
sampler_name: str = "Use same sampler"
sampler_name_override: Optional[str] = None