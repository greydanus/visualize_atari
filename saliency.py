# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings

warnings.filterwarnings("ignore")  # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2

# [210, 160, 3] -> [1, 80, 80]
prepro = (
    lambda img: cv2.resize(src=img[35:195].mean(2), dsize=(80, 80))
    .astype(np.float32)
    .reshape(1, 80, 80)
    / 255.0
)
searchlight = lambda im, mask: im * mask + gaussian_filter(im, sigma=3) * (
    1 - mask
)  # choose an area NOT to blur
occlude = (
    lambda im, mask: im * (1 - mask) + gaussian_filter(im, sigma=3) * mask
)  # choose an area to blur


def get_mask(center, size, r):
    y, x = np.ogrid[-center[0] : size[0] - center[0], -center[1] : size[1] - center[1]]
    keep = x * x + y * y <= 1
    mask = np.zeros(size)
    mask[keep] = 1  # select a circle of pixels
    mask = gaussian_filter(
        mask, sigma=r
    )  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask / mask.max()


def run_through_model(
    model,
    history,
    ix,
    interp_func=None,
    mask=None,
    blur_memory=None,
    mode="actor",
):
    # [210, 160, 3] -> [1, 80, 80]
    if mask is None:
        im = prepro(history["ins"][ix])
    else:
        assert interp_func is not None, "interp func cannot be none"
        # [210, 160, 3] -> [1, 80, 80]
        im = prepro(history["ins"][ix]).squeeze()
        # -> [1, 80, 80]
        im = interp_func(im, mask).reshape(1, 80, 80)  # perturb input im -> im'
    tens_state = torch.Tensor(im)
    state = tens_state.unsqueeze(0)
    hx = torch.tensor(history["hx"][ix - 1]).view(1, -1)
    cx = torch.tensor(history["cx"][ix - 1]).view(1, -1)
    if blur_memory is not None:
        cx.mul_(1 - blur_memory)  # perturb memory vector
    model_inp = (state, (hx, cx))
    if mode == "critic":
        return model(model_inp)[0]
    else:
        return model(model_inp)[1]


def score_frame(model, history, ix, r, d, interp_func, mode="actor"):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ["actor", "critic"], 'mode must be either "actor" or "critic"'
    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(80 / d) + 1, int(80 / d) + 1))  # saliency scores S(t,i,j)
    for i in range(0, 80, d):
        for j in range(0, 80, d):
            mask = get_mask(center=[i, j], size=[80, 80], r=r)
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode)
            scores[int(i / d), int(j / d)] = (L - l).pow(2).sum().mul_(0.5).item()
    pmax = scores.max()
    scores = cv2.resize(
        src=scores, dsize=(80, 80), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    return scores


def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = cv2.resize(
        src=saliency, dsize=(160, 160), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min()
    # S = fudge_factor * pmax * S / S.max()
    S = fudge_factor * S
    im = atari.astype("uint16")
    im[35:195, :, channel] += S.astype("uint16")
    im = im.clip(1, 255).astype("uint8")
    return im


def get_env_meta(env_name):
    meta = {}
    if env_name == "Pong-v0":
        meta["critic_ff"] = 600
        meta["actor_ff"] = 500
    elif env_name == "Breakout-v0":
        meta["critic_ff"] = 600
        meta["actor_ff"] = 300
    elif env_name == "SpaceInvaders-v0":
        meta["critic_ff"] = 400
        meta["actor_ff"] = 400
    else:
        print('environment "{}" not supported'.format(env_name))
    return meta
