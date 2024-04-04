import argparse
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import os
import random
import torch
from SAC import SAC_Agent
import seaborn as sns
import time
from torch.distributions import uniform
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

parser = argparse.ArgumentParser()
parser.add_argument('--eval_type', default='model',
                    choices=['model', 'model_noise'])
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='number of neurons in the hidden layers (default: 256)')
parser.add_argument('--EnvIdex', type=int, default=5, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2, Walker2D,Hopperv3')
parser.add_argument('--random_seed',type=int, default=None, help='random seed')
parser.add_argument('--method', default='mdp', choices=['mdp', 'pr_mdp', 'nr_mdp'])
parser.add_argument('--alpha1', type=float, default=0.1,help='control given to adversary (default: 0.1)')
parser.add_argument('--adaptive_step', type=str2bool, default=False, help='adaptive step size')
parser.add_argument('--b', type=float, default=0, help='ablation experience')
parser.add_argument('--ablation', type=str2bool, default=False, help='ablation experience')
# parser.add_argument('--run_num',type=int, default=2)
args = parser.parse_args()

def evaluate_policy(env, model, max_action, EnvIdex):
    scores = 0
    turns = 4
    # alpha = 0.2
    for j in range(turns):
        s, done, ep_r = env.reset(), False, 0
        while not done:
            # state attack
            # state = noise.sample(s.shape).view(s.shape)
            # state = state.numpy()
            # s = state + s
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True, with_logprob=False,mdp_type='mdp')
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            # action attack
            # if random.random() < alpha:
            #     act = noise.sample(a.shape).view(a.shape)
            #     act = act.numpy()
            s_prime, r, done, info = env.step(act)
            r = Reward_adapter(r, EnvIdex)
            ep_r += r
            s = s_prime
        # print(ep_r)
        scores += ep_r
    return scores/turns

EnvName = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v0','Humanoid-v2','HalfCheetah-v2','Walker2d-v3','Hopper-v3','Ant-v2','Swimmer-v2','InvertedDoublePendulum-v2']
BriefEnvName = ['BWv3', 'BWHv3', 'Lch_Cv2', 'PV0', 'Humanv2', 'HCv2','Walker','Hopper','Ant-v2','Swimmer-v2','IDPendulum-v2']
test_episodes = 8
env = gym.make(EnvName[args.EnvIdex])
random_seed = args.random_seed
if random_seed is None:
    random_seed = np.random.randint(0,4096)
env.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
max_action = float(env.action_space.high[0])
if args.adaptive_step:
    ad_dir = "adaptive_mdp"
else:
    ad_dir = args.method
agent = SAC_Agent(gamma=0.99,alpha1=0,state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],hid_shape=(args.hidden_size,args.hidden_size))
# noise = uniform.Uniform(-0.1,0.1)
# noise = uniform.Uniform(torch.tensor([-1.0]),torch.tensor([1.0]))
basic_bm = copy.deepcopy(env.env.model.body_mass.copy())
basic_gf = copy.deepcopy(env.env.model.geom_friction.copy())
if args.EnvIdex == 4:
    frictions = np.linspace(0.3,2.7,11,endpoint=True)
    masses = np.linspace(0.5,1.4,11,endpoint=True)
elif args.EnvIdex == 5:
    frictions = np.linspace(0.1,3.0,11,endpoint=True)
    masses = np.linspace(0.5,1.8,11,endpoint=True)
elif args.EnvIdex == 6:
    frictions = np.linspace(0.3,2.0,11,endpoint=True)
    masses = np.linspace(0.4,1.5,11,endpoint=True)
elif args.EnvIdex == 7:
    frictions = np.linspace(0.6,1.2,11,endpoint=True)
    masses = np.linspace(0.2,1.6,11,endpoint=True)
elif args.EnvIdex == 8:
    frictions = np.linspace(0.1,1.2,11,endpoint=True)
    masses = np.linspace(0.5,3.0,11,endpoint=True)
elif args.EnvIdex == 9:
    frictions = np.linspace(0,1.6,11,endpoint=True)
    masses = np.linspace(0.5,2.0,11,endpoint=True)
elif args.EnvIdex == 10:
    frictions = np.linspace(0.4,2.4,11,endpoint=True)
    masses = np.linspace(0.2,2.2,11,endpoint=True)
results = {}
for run_num in range(0,5):
    base_dir = os.getcwd() + '/models/' + EnvName[args.EnvIdex] + '/' + ad_dir + '_' + str(args.alpha1) + '/' + str(run_num)
    model_num = random.sample(range(1,10),4)
    for episode in model_num:
        agent.load(base_dir,episode)
        #agent.load(episode)
        # agent.eval()
        test_episodes = 8
        for friction in frictions: #np.linspace(0.8, 1.2, 20):
            if friction not in results:
                results[friction] = {}
            for idx in range(len(basic_gf)):
                env.env.model.geom_friction[idx] = basic_gf[idx] * friction
            for mass in masses:
                if mass not in results[friction]:
                    results[friction][mass]=[]
                ep_r = 0
                for idy in range(len(basic_bm)):
                    env.env.model.body_mass[idy] = basic_bm[idy] * mass
                for _ in range(test_episodes):
                    r = evaluate_policy(env, agent, max_action, args.EnvIdex)
                    ep_r += r
                results[friction][mass].append(ep_r/test_episodes)
    print("finish"+str(run_num))
if args.ablation:
    with open(os.getcwd() + '/test/choose_alpha/'+BriefEnvName[args.EnvIdex]+'/results_'+BriefEnvName[args.EnvIdex]+'_'+ad_dir+'ablation','wb') as f:
        pickle.dump(results, f)
else:
    with open(os.getcwd() + '/test/'+BriefEnvName[args.EnvIdex]+'/results_'+BriefEnvName[args.EnvIdex]+'_'+ad_dir+'_'+str(args.alpha1),'wb') as f:
        pickle.dump(results, f)
env.close()
results = open(os.getcwd() + '/test/'+BriefEnvName[args.EnvIdex]+'/results_'+BriefEnvName[args.EnvIdex]+'_'+ad_dir+'_'+str(args.alpha1),"rb")
results = pickle.load(results)
print("done1")
list1 = np.array([list(item.values()) for item in results.values()])
list1 = np.mean(list1,axis=2)
array = np.array(list1).reshape(11,11)
sns.set(font_scale=1.4)
f, ax = plt.subplots(figsize=(10,7))
if args.EnvIdex == 4:
    sns.heatmap(array,ax=ax,vmin=1900,vmax=4800,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
elif args.EnvIdex == 5:
    sns.heatmap(array,ax=ax,vmin=2500,vmax=9000,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
elif args.EnvIdex == 6:
    sns.heatmap(array,ax=ax,vmin=2000,vmax=4000,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
elif args.EnvIdex == 7:
    sns.heatmap(array,ax=ax,vmin=1300,vmax=3200,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
elif args.EnvIdex == 8:
    sns.heatmap(array,ax=ax,vmin=3000,vmax=5000,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
elif args.EnvIdex == 9:
    sns.heatmap(array,ax=ax,vmin=45,vmax=100,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
elif args.EnvIdex == 10:
    sns.heatmap(array,ax=ax,vmin=6000,vmax=10000,cmap='RdPu',annot=False,linewidths=2,cbar=True,cbar_kws={'label':'return',},)
EnvName = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v0','Humanoid','HalfCheetah','Walker2d','Hopper','Ant','Swimmer','InvertedDoublePendulum']

ax.set_title("Hopper SAC",fontsize=25)
ax.set_xlabel("relative mass")
ax.set_ylabel("relative friction")
frictions = np.around(frictions,2)
masses = np.around(masses,2)
ax.set_yticklabels(frictions)
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
ax.set_xticklabels(masses)
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
if args.ablation:
    plt.savefig(os.getcwd() + '/picture/choose_alpha/'+ad_dir+'_ablation.png')
else:
    plt.savefig(os.getcwd() + '/picture/'+BriefEnvName[args.EnvIdex]+'/'+ad_dir+'.png',bbox_inches='tight')
    plt.savefig(os.getcwd() + '/picture/' + BriefEnvName[args.EnvIdex] + '/' + ad_dir + '.pdf',bbox_inches='tight')
plt.show()
print("done2")
