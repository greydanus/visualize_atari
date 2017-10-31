# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import gym, sys
import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

sys.path.append('..')
from visualize_atari import *

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class OverfitAtari():
    def __init__(self, env_name, expert_dir, seed=0):
        self.atari = gym.make(env_name) ; self.atari.seed(seed)
        self.action_space = self.atari.action_space
        self.expert = NNPolicy(channels=1, num_actions=self.action_space.n)
        self.expert.try_load(expert_dir)
        self.cx = Variable(torch.zeros(1, 256)) # lstm memory vector
        self.hx = Variable(torch.zeros(1, 256)) # lstm activation vector
        
    def seed(self, s):
        self.atari.seed(s) ; torch.manual_seed(s)

    def reset(self):
        self.cx = Variable(torch.zeros(1, 256))
        self.hx = Variable(torch.zeros(1, 256))
        return self.atari.reset()

    def step(self, action):
        state, reward, done, info = self.atari.step(action)
        
        expert_state = torch.Tensor(prepro(state)) # get expert policy and incorporate it into environment
        _, logit, (hx, cx) = self.expert((Variable(expert_state.view(1,1,80,80)), (self.hx, self.cx)))
        self.hx, self.cx = Variable(hx.data), Variable(cx.data)
        
        expert_action = int(F.softmax(logit).data.max(1)[1][0,0])
        target = torch.zeros(logit.size()) ; target[0,expert_action] = 1
        j = 72 ; k = 5
        expert_action = expert_action if False else np.random.randint(self.atari.action_space.n)
        for i in range(self.atari.action_space.n):
            state[37:41, j + k*i: j+1+k*i,:] = 250 if expert_action == i else 50
        return state, reward, done, target