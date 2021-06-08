# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ravens main training script."""

import os
import pickle

from absl import app
from absl import flags
import numpy as np
from ravens import agents
from ravens import dataset
from ravens import tasks
from ravens.environments.environment import Environment
import tensorflow as tf
# Softgym imports
from PIL import Image
import copy
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.visualization import save_numpy_as_gif
from envs.env import Env

flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('assets_root', './assets/', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'hanoi', '')
flags.DEFINE_string('agent', 'transporter', '')
flags.DEFINE_integer('n_demos', 100, '')
flags.DEFINE_integer('n_steps', 40000, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
# Softgym Flags TODO



FLAGS = flags.FLAGS


def main(unused_argv):
  # TODO recover args from demo 

  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')

  # Configure how much GPU to use (in Gigabytes).
  if FLAGS.gpu_limit is not None:
    mem_limit = 1024 * FLAGS.gpu_limit
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)
  
  # =============================
  # Initialize Softgym environment.
  # =============================
  env_name = 'ClothFoldPPP' # 'ClothFoldPPP'
  env_kwargs = env_arg_dict[env_name]  # Default env parameters

  env_kwargs['use_cached_states'] = False
  env_kwargs['save_cached_states'] = False
  env_kwargs['num_variations'] = 1
  env_kwargs['render'] = True
  env_kwargs['headless'] = False

  env_class = Env  
  env = SOFTGYM_ENVS[env_name](**env_kwargs)

  # Recover trajectories from dict (TODO:automate getting same traj used for training)
  handle = open(./cem/trajs/cem_traj_test.pkl', 'rb')
  traj_dict = pickle.load(handle)
  print("Reading trajectories from :", handle)

  initial_states = traj_dict['initial_states']        
  #action_trajs = traj_dict['action_trajs']
  #configs = traj_dict['configs']
  # =============================


  # Load test dataset.
  ds = dataset.Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-test'))

  # Run testing for each training run.
  for train_run in range(FLAGS.n_runs):
    name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'

    # Initialize agent.
    np.random.seed(train_run)
    tf.random.set_seed(train_run)
    agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)

    # # Run testing every interval.
    # for train_step in range(0, FLAGS.n_steps + 1, FLAGS.interval):

    # Load trained agent.
    if FLAGS.n_steps > 0:
      agent.load(FLAGS.n_steps)

    # Run testing and save total rewards with last transition info.
    results = []
    for i in range(ds.n_episodes):
      print(f'Test: {i + 1}/{ds.n_episodes}')
      episode, seed = ds.load(i)
      goal = episode[-1] # Goal for Fold
      #goal = episode[0]
      total_reward = 0
      #np.random.seed(seed)
      #env.seed(seed) # Seed overwrites initial config?

      # Reset env and load obs
      env.reset(initial_state=initial_states[0])
      color = env.get_image(128, 128) # Only supports 128, 128 TODO.
      depth = env.get_depth_map()
      obs = {'color': color , 'depth': depth}
      info = None
      reward = 0.0
      frames = [color]
      for _ in range(20): # TODO remove hardcoded len, this matches cem horizon for ClothFoldPPP
        act = agent.act(obs, None, goal)

        img = Image.fromarray(obs['color'], 'RGB')
        #img.show() # DEBUG

        act0 = np.zeros(shape=(4,))
        act1 = np.zeros(shape=(4,))
        act0[:3] = act['pose0'][0] # Take (x, y, z) form pick
        act1[:3] = act['pose1'][0] # Take (x, y, z) form place
        act0[3] = 1.0 # In sg action space act[3] > 0.5 is pick
        act1[3] = 0.1 # In sg action space act[3] < 0.5 is place
        env.step(act0)
        _, reward, done, info = env.step(act1)
        reward = info['normalized_performance']
        total_reward += reward

        color = env.get_image(128, 128) 
        depth = env.get_depth_map()
        #print("Color and depth", color.shape, depth.shape, np.amax(depth))
        obs = {'color': color , 'depth': depth}
        frames.append(color)
        print(f'Total Reward: {total_reward} Done: {done}')
        if done:
          break
      print("Final episode normalized performance: ", reward)
      results.append((total_reward, info))

      # Save video
      gif_name = 'gif_test.gif'
      save_numpy_as_gif(np.array(frames), gif_name)

      # Save results.
      with tf.io.gfile.GFile(
          os.path.join(FLAGS.root_dir, f'{name}-{FLAGS.n_steps}.pkl'),
          'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
  app.run(main)
