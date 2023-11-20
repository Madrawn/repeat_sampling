from lib_repeat_sampling import global_state


def override_repeat_denoise_strength(repeat_denoise_strength: float):
    global_state.repeat_denoise_strength_override = repeat_denoise_strength
