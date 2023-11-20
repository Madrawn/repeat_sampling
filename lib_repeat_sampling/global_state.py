from typing import List, Optional


is_enabled: bool = False
repeat_denoise_strength: float = 0.0
verbose: bool = True
repeat_denoise_strength_override: Optional[float] = None
checkbox: bool
repeat_denoise_strength: float
repeats: int
tactic: str
factor: float


def apply_and_clear_repeat_denoise_strength_override():
    global repeat_denoise_strength, repeat_denoise_strength_override
    if repeat_denoise_strength_override is not None:
        repeat_denoise_strength = repeat_denoise_strength_override
        repeat_denoise_strength_override = None
