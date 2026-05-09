import numpy as np
from scipy.signal import savgol_filter


def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x

# def derivative_of(x, dt=0.4, radian=False):
#     if radian:
#         x = make_continuous_copy(x)

#     valid_mask = ~np.isnan(x)
#     if valid_mask.sum() < 2:
#         return np.zeros_like(x)

#     valid_x = x[valid_mask]
#     valid_t = np.where(valid_mask)[0] * dt 
    
#     dx = np.full_like(x, np.nan)
#     dx[valid_mask] = np.gradient(valid_x, valid_t)

#     return dx

def derivative_of(x, dt=1, radian=False, smooth = True):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    if smooth and len(masked_x) >= 5:
        # window_length 必须是奇数，polyorder 通常选 2 或 3
        # 窗口大小选 5 或 7 比较适合 2.5 FPS 的数据
        masked_x = savgol_filter(masked_x, window_length=5, polyorder=2)
        
    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx