import torch
from scipy import integrate

from ...util import append_dims


def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):

    d = {
        'video_d': (x['noised_video_input'] - denoised['video_output']) / append_dims(sigma['video_sigmas'], x['noised_video_input'].ndim),
        'pose_d': (x['noised_pose_input'] - denoised['pose_output']) / append_dims(sigma['pose_sigmas'], x['noised_pose_input'].ndim),
    }

    return d


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
