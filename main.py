import math

import numpy as np
import torch
import gym
from SAC import SAC_Agent
from ReplayBuffer import RandomBuffer, device
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os, shutil
import argparse
from Adapter import *

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=5, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2, Walker2D,Hopperv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=2250000, help='which model to load')
parser.add_argument('--seed', type=int, default=None, help='random seed')

parser.add_argument('--total_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model saving interval, in steps.')
parser.add_argument('--save_interval', type=int, default=int(1e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_turn', type=int, default=3, help='Model evaluating times, in episode.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--adv_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--alpha1', type=float, default=0.1,help='control given to adversary (default: 0.1)')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--method', default='mdp', choices=['mdp', 'pr_mdp', 'nr_mdp'])
parser.add_argument('--adaptive_step', type=str2bool, default=False, help='adaptive step size')
parser.add_argument('--choose_alpha', type=str2bool, default=False, help='ablation alpha')
parser.add_argument('--ablation', type=str2bool, default=False, help='ablation')
opt = parser.parse_args()
print(opt)


def evaluate_policy(env, model, render, steps_per_epoch, max_action, EnvIdex,mdp_type):
    scores = 0
    turns = opt.eval_turn
    for j in range(turns):
        s, done, ep_r = env.reset(), False, 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True, with_logprob=False,mdp_type=mdp_type)
            a = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            # act = np.concatenate(
            #     (np.array(np.full(env.adv_action_space.shape[0], np.nan)),a),axis=0) 
            s_prime, r, done, info = env.step(a)
            r = Reward_adapter(r, EnvIdex)
            ep_r += r
            s = s_prime
            if render:
                env.render()
        # print(ep_r)
        scores += ep_r
    return scores/turns

def main():

    write = opt.write   #Use SummaryWriter to record the training.
    render = opt.render

    # Env config:
    EnvName = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v0','Humanoid-v2','HalfCheetah-v2','Walker2d-v3','Hopper-v3','Ant-v2','Swimmer-v2','InvertedDoublePendulum-v2']
    BriefEnvName = ['BWv3', 'BWHv3', 'Lch_Cv2', 'PV0', 'Humanv2', 'HCv2','Walker2d','Hopperv3','Ant-v2','Swimmerv2','IDPendulum-v2']
    Env_With_Dead = [True, True, True, False, True, False, True,True,True,False,True]
    EnvIdex = opt.EnvIdex
    env_with_Dead = Env_With_Dead[EnvIdex]
    #Env like 'LunarLanderContinuous-v2' is with Dead Signal. Important!
    #Env like 'Pendulum-v0' is without Dead Signal.
    env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make(EnvName[EnvIdex])
    adversary_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    steps_per_epoch = env._max_episode_steps
    print('Env:',EnvName[EnvIdex],'  state_dim:',state_dim,'  action_dim:',action_dim,
          '  max_a:',max_action,'  min_a:',env.action_space.low[0], 'max_episode_steps:', steps_per_epoch,"device:",device)

    #Interaction config:
    start_steps = 5*steps_per_epoch #in steps
    update_after = 2*steps_per_epoch #in steps
    update_every = opt.update_every
    total_steps = opt.total_steps
    eval_interval = opt.eval_interval
    save_interval = opt.save_interval
    mdp_type = opt.method
    #Random seed config:
    random_seed = opt.seed
    if random_seed is None:
        random_seed = np.random.randint(0,4096)
    print("Random Seed: {}".format(random_seed))
    np.random.seed(random_seed)
    #random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.seed(random_seed)
    eval_env.seed(random_seed)
    if opt.choose_alpha:
        if opt.EnvIdex == 4 or opt.EnvIdex == 5:
            num = np.random.randint(0,10)
            opt.alpha1 = 0.05+0.01*num
        elif opt.EnvIdex == 6:
            num = np.random.randint(0,15)
            opt.alpha1 = 0.03+0.01*num
        elif opt.EnvIdex == 7:
            num = np.random.randint(0,15)
            opt.alpha1 = 0.03+0.01*num
    if opt.adaptive_step:
        adaptive_step = True
        ad_dir = "adaptive_mdp"
    else:
        adaptive_step = False
        ad_dir = opt.method
    if opt.choose_alpha:
        base_dir = os.getcwd() + '/models/choose_alpha/' + EnvName[EnvIdex] + '/'+ad_dir + '_' +str(opt.alpha1) + '/'
    else:
        base_dir = os.getcwd() + '/models/' + EnvName[EnvIdex] + '/'+ad_dir + '_' +str(opt.alpha1) + '/'
    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)
    #SummaryWriter config:
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = '_'+timenow[0:10]+'_'+timenow[11:13]+'_'+timenow[-2::]
        writepath = 'runs/'+ EnvName[EnvIdex] +'/SAC_{}'.format(BriefEnvName[EnvIdex]) + timenow+ '_'+ad_dir+str(opt.alpha1)+ '_'+str(run_number)
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    #Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width,opt.net_width),
        "a_lr": opt.a_lr,
        "adv_lr":opt.adv_lr,
        "c_lr": opt.c_lr,
        "batch_size":opt.batch_size,
        "alpha":opt.alpha,
        "alpha1":opt.alpha1,
        "adaptive_alpha":opt.adaptive_alpha
    }
    value_losses = []
    policy_losses = []
    adversary_losses = []
    adv_adjust_steps = []
    gradient_norms = []
    model_num = 0
    model = SAC_Agent(**kwargs)

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_Dead, max_size=int(1e6))
    if render:
        average_reward = evaluate_policy(env, model, render, steps_per_epoch, max_action, EnvIdex)
        print('Average Reward:', average_reward)
    else:
        s, done, current_steps = env.reset(), False, 0
        train_steps = 0
        for t in range(total_steps):
            current_steps += 1
            '''Interact & trian'''

            if t < start_steps:
                #Random explore for start_steps
                act = env.action_space.sample() #act∈[-max,max]
                a = Action_adapter_reverse(act,max_action) #a∈[-1,1]
                # act = np.concatenate(
                #     (np.array(np.full(env.adv_action_space.shape[0], np.nan)),a),axis=0)
            else:
                a = model.select_action(s, deterministic=False, with_logprob=False,mdp_type=opt.method,adaptive_step=opt.adaptive_step) #a∈[-1,1]
                act = Action_adapter(a,max_action) #act∈[-max,max]
                # act = np.concatenate(
                #     (np.array(np.full(env.adv_action_space.shape[0], np.nan)),act),axis=0)
            s_prime, r, done, info = env.step(act)
            dead = Done_adapter(r, done, current_steps, EnvIdex)
            r = Reward_adapter(r, EnvIdex)
            replay_buffer.add(s, a, r, s_prime, dead)
            s = s_prime


            # 50 environment steps company with 50 gradient steps.
            # Stabler than 1 environment step company with 1 gradient step.
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    adversary_update = (train_steps % 10 == 0 )
                    value_loss, policy_loss, adversary_loss, adv_adjust_step, gradient_norm = model.train(replay_buffer,mdp_type,adversary_update,adaptive_step)
                    value_losses.append(value_loss)
                    policy_losses.append(policy_loss)
                    adversary_losses.append(adversary_loss)
                    adv_adjust_steps.append(adv_adjust_step)
                    gradient_norms.append(gradient_norm)
                    train_steps += 1
                if write:
                    writer.add_scalar('value_losses',np.mean(value_losses),global_step=t + 1)
                    writer.add_scalar('policy_losses',np.mean(policy_losses),global_step=t + 1)
                    writer.add_scalar('adversary_losses',np.mean(adversary_losses),global_step=t + 1)
                    writer.add_histogram('adv_alpha',np.mean(adv_adjust_steps),global_step=t+1)
                    writer.add_histogram('gradient_norms',np.mean(gradient_norms),global_step=t+1)
                del value_losses[:]
                del policy_losses[:]
                del adversary_losses[:]
                del adv_adjust_steps[:]
                del gradient_norms[:]
            '''adjust distribution'''

            '''save model'''
            if (t+1) >= (total_steps-9000) and (t + 1) % save_interval == 0:
                model_num += 1
                model.save(base_dir,model_num)

            '''record & log'''
            if (t + 1) % eval_interval == 0:
                with torch.no_grad():
                    score = evaluate_policy(eval_env, model, False, steps_per_epoch, max_action, EnvIdex,mdp_type='mdp')
                    writer.add_scalar('ep_r', score, global_step=t + 1)
                    writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                print('EnvName:', EnvName[EnvIdex], 'seed:', random_seed, 'totalsteps:', t+1, 'score:', score)
            if done:
                s, done, current_steps = env.reset(), False, 0

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()