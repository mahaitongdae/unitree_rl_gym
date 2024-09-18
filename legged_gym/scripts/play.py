import pathlib
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch
import onnxruntime as ort
import datetime

onnx_model_path = '/home/naliseas-workstation/Documents/haitong/unitree_rl_gym/logs/models/policy_50000.onnx'
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

def get_action(input_data):

    if not isinstance(input_data, np.ndarray):
        raise ValueError("Input data must be a numpy array")

    # Run the model on the input data
    output = session.run(None, {input_name: input_data})

    # Return the results
    return output[0]


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        path = pathlib.Path(ppo_runner.load_path)
        # export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_model_path = os.path.join(path.parent, 'exported')
        run_name = str(path).split('/')[-2]
        checkpoint_num = str(path).split('model_')[1][:-3]
        os.makedirs(export_model_path, exist_ok=True)
        fname = datetime.datetime.now().strftime(f'policy_{run_name}{checkpoint_num}_%m%d-%H%M.onnx')
        export_policy_as_onnx(ppo_runner.alg.actor_critic,
                              export_model_path,
                              filename=fname)
        print('Exported policy as jit script to: ', export_model_path + fname)
    # last_contact = 
    # contact_changes = torch.zeros_like(last_contact)
    # last_change = 0
    count = torch.zeros_like(obs[:, 44:])
    lin_speed = []

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        # actions = torch.zeros([obs.shape[0], 12])
        # actions = get_action(obs.detach().cpu().numpy())#  / 0.25
        # actions = torch.zeros_like(actions)
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        # contact = env.get_contact()
        # print(torch.ones_like(contact) - contact)
        # last_contact = contact
        # count += (torch.ones_like(contact.float()) - contact.float())
        lin_speed.append((obs[:, 0] / 2).mean())

    # print((count / env.max_episode_length).mean())
    # print((count / env.max_episode_length).std())
    # print(torch.mean(torch.tensor(lin_speed)), torch.std(torch.tensor(lin_speed)),)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
