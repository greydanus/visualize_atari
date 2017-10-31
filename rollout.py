# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def rollout(model, env, max_ep_len=3e3, render=False):
    history = {'ins': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
    
    state = torch.Tensor(prepro(env.reset())) # get first state
    episode_length, epr, eploss, done  = 0, 0, 0, False # bookkeeping
    hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        value, logit, (hx, cx) = model((Variable(state.view(1,1,80,80)), (hx, cx)))
        hx, cx = Variable(hx.data), Variable(cx.data)
        prob = F.softmax(logit)

        action = prob.max(1)[1].data # prob.multinomial().data[0] # 
        obs, reward, done, expert_policy = env.step(action.numpy()[0])
        if render: env.render()
        state = torch.Tensor(prepro(obs)) ; epr += reward

        # save info!
        history['ins'].append(obs)
        history['hx'].append(hx.squeeze(0).data.numpy())
        history['cx'].append(cx.squeeze(0).data.numpy())
        history['logits'].append(logit.data.numpy()[0])
        history['values'].append(value.data.numpy()[0])
        history['outs'].append(prob.data.numpy()[0])
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history
