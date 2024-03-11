# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
import numpy as np

import pdb
import sys
import pathlib
import ruamel.yaml as yaml

# import argparse
from attrdict import AttrDict

from model import *

from env import Env
from dreamer.models import *
from dreamer import tools
  

class DreamerWrapper:
  def __init__(self, env:Env, args):
    
    self._env = env
    # if wrapper_args:
    #   self.wrapper_args = wrapper_args
    # else:
    #   self.wrapper_args = WrapperCfg()
    self.num_actions = self._env.action_space()
    self.update_interval = 5
    self.global_counter = 0
    self._policy_encoder = AtariEncoder(args) # TODO: Test what if using the same encoder as dreamer
    self._build_world_model()
    self._init_world_model_dataset()
    
  def _build_world_model(self,headless=True,sim_device='cuda:0'):
    # TODO: use dreamer_args in this function.
    # world model
    print('Begin construct world model')
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "dreamer/configs.yaml").read_text()
    ) # NOTE: Change this to the correct path

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--headless", action="store_true", default=False)
    # parser.add_argument("--sim_device", default='cuda:0')
    # for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        # arg_type = tools.args_type(value)
        # parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    # self.wm_config = parser.parse_args()
    defaults["headless"] = headless
    defaults["sim_device"] = sim_device
    self.wm_config = AttrDict(defaults)
    
    # allow world model and env & alg on different device
    # self.wm_config.device = self.wm_config.sim_device
    self.wm_config.num_actions = self.wm_config.num_actions * self.update_interval
    # prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions
    # image_shape = self.env.cfg.depth.resized + (1,)
    image_shape = (84,84,1)
    obs_shape = {'image': image_shape}

    self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=True)
    self._world_model = self._world_model.to(self._world_model.device)
    print('Finish construct world model')
    
    self.wm_feature_dim = self.wm_config.dyn_deter + self.wm_config.dyn_stoch * self.wm_config.dyn_discrete
    

  def _init_world_model_dataset(self):
    # init world model input
    self.step_in_wm_dataset = 0
    
    self.wm_is_first = torch.scalar_tensor(False,device=self._world_model.device)
    
    self.wm_obs = {
      "image": torch.zeros((84,84,self._env.window), device=self._world_model.device),
      "is_first": self.wm_is_first
    }


    # wm_metrics = None
    self.wm_action_history = torch.zeros(size=(self.update_interval, self.num_actions),
                                    device=self._world_model.device)
    # wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
    # wm_feature = torch.zeros((self.wm_feature_dim))

    max_episode_length = self._env.ale.getInt('max_num_frames_per_episode') # NOTE: change by _env
    wm_dataset_length = int(max_episode_length / self.update_interval) + 3
    
    self.wm_dataset = {
        "image": torch.zeros((wm_dataset_length, self._env.window, 84,84), device=self._world_model.device),
        "action": torch.zeros((wm_dataset_length,
                                self.num_actions * self.update_interval), device=self._world_model.device),
        "reward": torch.zeros((wm_dataset_length),
                              device=self._world_model.device),
    }

    self.wm_dataset_size = 0

    self.wm_buffer = {
        "image": torch.zeros(
            (wm_dataset_length,self._env.window,84,84),
            device=self._world_model.device),
        "action": torch.zeros((wm_dataset_length,
                                self.num_actions * self.update_interval), device=self._world_model.device),
        "reward": torch.zeros((wm_dataset_length),
                              device=self._world_model.device),
    }

    self.wm_buffer_index = 0

  # def _get_world_model_feat(self):
  #   wm_obs = self.wm_obs
    
  #   wm_action = self.wm_action_history.flatten(1)
    
  #   # use world model to interface (every 5 steps)
  #   if (self.env.global_counter % self.wrapper_args.update_interval == 0):
  #     # world model obs step
  #     wm_embed = self._world_model.encoder(wm_obs)
  #     wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
  #                                                         wm_obs["is_first"])
  #     wm_feature = self._world_model.dynamics.get_feat(wm_latent)
  #     # wm_is_first[:] = 0
  #     return wm_embed, wm_feature
    
  def _get_world_model_feat(self,wm_obs, wm_action,wm_is_first):
    # world model obs step
    wm_embed = self._world_model.encoder(wm_obs)
    wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                        wm_is_first)
    wm_feature = self._world_model.dynamics.get_feat(wm_latent)
    # wm_is_first[:] = 0
    return wm_feature
  
  def update_wm_buffer(self,state,action,reward,done):
    self.wm_obs = {
      "image": state.to(self._world_model.device),
      "is_first": self.wm_is_first
    }
    
    action = torch.zeros(self.num_actions, device=self._world_model.device).scatter(0, torch.tensor(action).to(self._world_model.device), 1) # action: int to one-hot
    self.wm_action_history = torch.concat(
          (self.wm_action_history[1:], action.unsqueeze(0)), dim=0)
    
    self.wm_is_first = False
    
    # pdb.set_trace()
    self.wm_buffer["image"][self.wm_buffer_index] = state
    self.wm_buffer["action"][self.wm_buffer_index] = self.wm_action_history.flatten()
    self.wm_buffer["reward"][self.wm_buffer_index] = reward
    self.wm_buffer_index += 1
    
    if done:
      self.wm_buffer_index = 0
      self.wm_dataset_size = min(self.wm_dataset_size+1, self.wm_dataset_length)
      self.wm_dataset["image"][:self.wm_dataset_size] = self.wm_buffer["image"]
      self.wm_dataset["action"][:self.wm_dataset_size] = self.wm_buffer["action"]
      self.wm_dataset["reward"][:self.wm_dataset_size] = self.wm_buffer["reward"]
      self.step_in_wm_dataset = self.wm_dataset_size
      self.wm_is_first = True
  
  def reset(self):
  
    policy_state =  self._env.reset()
    self.wm_is_first = True
    self.wm_obs = {
      "image":policy_state.to(self._world_model.device),
      "is_first": self.wm_is_first
    }
    self.wm_action_history = torch.zeros_like(self.wm_action_history)
    
    wm_feat = self._get_world_model_feat(self.wm_obs, self.wm_action_history.flatten(1))
    
    return torch.cat((self._policy_encoder(policy_state), wm_feat), dim=1)
    
  def step(self, action):      
    # get world model input
    state, reward, done = self._env.step(action)
    
    # update world model buffer
    self.update_wm_buffer(state,action,reward,done)

    policy_state = self._policy_encoder(state)
    wm_feat = self._get_world_model_feat(self.wm_obs, self.wm_action_history.flatten(1))
    policy_state = torch.cat((policy_state, wm_feat), dim=1)
    
    # update action history
    info = {}
    # info["action"] = self.wm_action_history.flatten(1)
    
    self.global_counter += 1
    # return policy_state, reward, done, info
    return policy_state, reward, done, info
  
  def learn_world_model(self, it):
    # Train World Model
    # start_time = time.time()
    
    if (self.step_in_wm_dataset > self.wm_config.train_start_steps):
        wm_metrics = {}
        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                          p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
            if (batch_length <= 1):
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((self.wm_config.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            post, context, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        
        for name, values in wm_metrics.items():
            self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
    # print('train world model time:', time.time() - start_time)

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True
    self._world_model.train()

  # Uses standard terminal signal
  def eval(self):
    self.training = False
    self._world_model.eval()

  def action_space(self):
    return self._env.action_space()

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()
