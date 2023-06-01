import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import gfootball.env as football_env
import time, pprint, json, os, importlib, shutil
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter
from train import save_args, copy_models
from actor import *
from learner import *
from evaluator import evaluator
from datetime import datetime, timedelta
import gfootball_engine as libgame
from PIL import Image

class Demo(object):
    def __init__(self, model) -> None:
        self.env0 = football_env.create_environment(env_name='11_vs_11_hard_stochastic', representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
        self.env1 = football_env.create_environment(env_name='11_vs_11_hard_stochastic', representation="pixels", stacked=False, logdir='/tmp/football', \
                                        write_goal_dumps=False, write_full_episode_dumps=False, render=True)
        self.model = model
        self.img_list = []


    def set_formation(self, formation):
        if formation == "433":
            self.env0.close()
            self.env1.close()
            self.env0 = football_env.create_environment(env_name='11_vs_11_hard_stochastic_433', representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
            self.env1 = football_env.create_environment(env_name='11_vs_11_hard_stochastic_433', representation="pixels", stacked=False, logdir='/tmp/football', \
                                        write_goal_dumps=False, write_full_episode_dumps=False, render=True)
        elif formation == "4231":
            self.env0.close()
            self.env1.close()
            self.env0 = football_env.create_environment(env_name="11_vs_11_hard_stochastic_4231", representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
            self.env1 = football_env.create_environment(env_name="11_vs_11_hard_stochastic_4231", representation="pixels", stacked=False, logdir='/tmp/football', \
                                        write_goal_dumps=False, write_full_episode_dumps=False, render=True) 
        elif formation == "442": 
            self.env0.close()
            self.env1.close()
            self.env0 = football_env.create_environment(env_name="11_vs_11_hard_stochastic_442", representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
            self.env1 = football_env.create_environment(env_name="11_vs_11_hard_stochastic_442", representation="pixels", stacked=False, logdir='/tmp/football', \
                                        write_goal_dumps=False, write_full_episode_dumps=False, render=True)
        else:
            raise Exception("输入的阵容错误！")


    def env_reset(self):
        self.env0.close()
        self.env1.close()
        self.env0 = football_env.create_environment(env_name='11_vs_11_hard_stochastic', representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
        self.env1 = football_env.create_environment(env_name='11_vs_11_hard_stochastic', representation="pixels", stacked=False, logdir='/tmp/football', \
                                        write_goal_dumps=False, write_full_episode_dumps=False, render=True) 


    def env_step(self, obs, h_out):
        h_in = h_out
        state_dict = fe.encode(obs[0])
        state_dict_tensor = state_to_tensor(state_dict, h_in)
        with torch.no_grad():
            a_prob, m_prob, _, h_out = self.model(state_dict_tensor)

        real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
        obs, rew, done, info = self.env0.step(real_action)
        obs_p, _, _, _ = self.env1.step(real_action)
        self.img_list.append(obs_p)
        return obs_p, done, obs
    
    def get_start_img(self):
        obs_p, _, _, _ = self.env1.step(0)
        self.img_list.append(obs_p)
        return obs_p


if __name__ == "__main__":
    arg_dict = {
    "env": "11_vs_11_kaggle",
    # "11_vs_11_kaggle" : environment used for self-play training
    # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
    "num_processes": 50,  # should be less than the number of cpu cores in your workstation.
    "batch_size": 32,   
    "buffer_size": 6,
    "rollout_len": 30,

    "lstm_size" : 256,
    "k_epoch" : 3,
    "learning_rate" : 0.0001,
    "gamma" : 0.993,
    "lmbda" : 0.96,
    "entropy_coef" : 0.0001,
    "grad_clip" : 3.0,
    "eps_clip" : 0.1,

    "summary_game_window" : 10, 
    "model_save_interval" : 300000,  # number of gradient updates bewteen saving model

    "trained_model_path" : None, #'/home/logs/[04-01]12.23.27/model_57018240.tar', # None use when you want to continue traning from given model.
    "latest_ratio" : 0.5, # works only for self_play training. 
    "latest_n_model" : 10, # works only for self_play training. 
    "print_mode" : False,

    "encoder" : "encoder_slide",
    "rewarder" : "rewarder_basic",
    "model" : "conv1d",
    "algorithm" : "ppo",
    "representation": 'simple115v2',
    "opponent_path": "./kaggle_simulations/agent/6th",
    "env_evaluation": '11_vs_11_hard_stochastic'  # for evaluation of self-play trained agent (like validation set in Supervised Learning)
    }
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cur_time = datetime.now() + timedelta(hours = 9)
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    save_args(arg_dict)
    if arg_dict["trained_model_path"] and 'kaggle' in arg_dict['env']: 
        copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir'])

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)
    
    fe = importlib.import_module("encoders." + arg_dict["encoder"])
    fe = fe.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()
    model = importlib.import_module("models." + arg_dict["model"])
    cpu_device = torch.device('cpu')
    center_model = model.Model(arg_dict)
    
    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
        optimization_step = checkpoint['optimization_step']
        center_model.load_state_dict(checkpoint['model_state_dict'])
        center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        arg_dict["optimization_step"] = optimization_step
        print("Trained model", arg_dict["trained_model_path"] ,"successfully loaded") 
    else:
        optimization_step = 0

    model_dict = {
        'optimization_step': optimization_step,
        'model_state_dict': center_model.state_dict(),
        'optimizer_state_dict': center_model.optimizer.state_dict(),
    }
    
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
    
    fe = fe_module.FeatureEncoder()


    demo = Demo(center_model)
    obs = demo.env0.observation()
    step = 1
    while True:
        
        # TODO 推流
        # TODO 获取指令instruction
        # reset 表示重置环境 set_formation 表示设置阵容,img为返回的图片（是numpy格式）
        instruction = 
        if isinstance(instruction, str):
            if instruction == 'reset':
                demo.env_reset()
                img = demo.get_start_img()
                obs = demo.env0.observation()
            elif "set_formation" in instruction:
                if "442" in instruction:
                    demo.set_formation("442")
                    img = demo.get_start_img()
                if "4231" in instruction:
                    demo.set_formation("4231")
                    img = demo.get_start_img()
                if "433" in instruction:    
                    demo.set_formation("433")   
                    img = demo.get_start_img() 
            else:
                raise ValueError(f"Error instruction: {instruction}")
        img, done, obs = demo.env_step(obs, h_out)
        if done:
            break
    
    print(len(demo.img_list))
    # img = demo.img_list[0]
    # im = Image.fromarray(img)
    # im.save("formation.jpeg")
    # img = Image.fromarray(demo.img_list[0])
    # img.save("formation.jepg")
    # input()
