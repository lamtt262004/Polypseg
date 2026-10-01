import copy

import numpy as np


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def generate_model(model, ema=False):
    model_copy = copy.deepcopy(model)
    if ema:
        for param in model_copy.parameters():
            param.detach_()
            param.requires_grad_(False)
    return model_copy


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
