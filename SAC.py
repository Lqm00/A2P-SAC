import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from torch.autograd import grad
from torch import optim, norm
from ReplayBuffer import device


def build_net(layer_shape, activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor, self).__init__()

        layers = [state_dim] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def calcu_prob(self,state,action):
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)  # 由mu和std参数化的正态分布（高斯）
        # u = dist.rsample()
        # a = torch.tanh(u)
        u = torch.atanh(action)
        # print(u)
        logp_pi_a = dist.log_prob(u)
        # print(logp_pi_a)
        return logp_pi_a

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)  # 由mu和std参数化的正态分布（高斯）

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # '''reparameterization trick of Gaussian'''#rsample是先对标准正态分布进行取样，然后输出：mu+std*采样值,重参数化是为了传导梯度
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = 0

        return a, logp_pi_a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


class SAC_Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            hid_shape=(256, 256),
            a_lr=3e-4,
            adv_lr=3e-4,
            c_lr=3e-4,
            batch_size=256,
            alpha=0.2,
            alpha1=0.1,
            adaptive_alpha=True
    ):
        self.actor = Actor(state_dim, action_dim, hid_shape).to(device)
        self.actor_perturbed = Actor(state_dim, action_dim, hid_shape).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.adversary = Actor(state_dim, action_dim, hid_shape).to(device)
        self.adversary_perturbed = Actor(state_dim, action_dim, hid_shape).to(device)
        self.adversary_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=adv_lr)
        self.q_critic = Q_Critic(state_dim, action_dim, hid_shape).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = 0.005
        self.batch_size = batch_size
        self.a1 = 0
        # self.adv_step_adjust = torch.tensor([1],requires_grad=False).to(device)
        self.a2 = 0
        # self.adv_step_sample = torch.tensor([1],requires_grad=False).to(device)
        self.a3 = 0
        # self.adv_step_critic = torch.tensor([1],requires_grad=False).to(device)
        self.a4 = 0
        self.alpha = alpha
        self.alpha1 = alpha1
        self.alpha2 = alpha1
        self.alpha3 = alpha1
        self.alpha4 = alpha1
        self.adaptive_alpha = adaptive_alpha
        self.noise_weights_func = self._get_uniform_weights
        if adaptive_alpha:
            self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)

    def _get_uniform_weights(self, size):
        return torch.ones(*size, device=device)

    def select_action(self, state, deterministic, with_logprob=False, mdp_type='mdp', adaptive_step=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if mdp_type != 'mdp':
            if mdp_type == 'nr_mdp':
                if adaptive_step:
                    with torch.no_grad():
                        a, _ = self.actor(state, deterministic, with_logprob)
                    adv_a, _ = self.adversary(state, deterministic, False)
                    adv_a = adv_a
                    with torch.no_grad():
                        norm_a = torch.norm(a - adv_a, p=2, dim=1)
                        norm_a = norm_a.mean()
                        self.adjust_step_sample(norm_a)
                    a = a.data * (1 - self.alpha2)
                    adv_a = adv_a * self.alpha2
                    a += adv_a
                else:
                    # self.alpha1 = random.uniform(0.03,0.18)
                    a, _ = self.actor(state, deterministic, with_logprob)
                    a = a.data * (1 - self.alpha1)
                    adv_a, _ = self.adversary(state, deterministic, False)

                    adv_a = adv_a.data * self.alpha1
                    a += adv_a
            elif mdp_type == 'pr_mdp':
                if np.random.rand() < (1 - self.alpha1):
                    a, _ = self.actor(state, deterministic, with_logprob)
                    a = a.data
                else:
                    a, _ = self.adversary(state, deterministic, False)
                    a = a.data
            else:
                a, _ = self.adversary(state, deterministic, False)
                a = a.data
        else:
            a, _ = self.actor(state, deterministic, with_logprob)
            a = a.data
        return a.cpu().detach().numpy().flatten()

    def update_robust(self, replay_buffer, adversary_update, mdp_type, robust_update, adaptive_step):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)
        # Train critic
        adv_adjust_step = 0
        gradient_norm = 0
        if robust_update == 'full':
            if mdp_type == 'nr_mdp':
                if adaptive_step:
                    with torch.no_grad():
                        adv_prime, _ = self.adversary(s_prime, False, False)
                        adv_prime = adv_prime
                    a_prime, log_pi_a_prime = self.actor(s_prime, False, True)
                    with torch.no_grad():
                        norm_a = torch.norm(a_prime - adv_prime, p=2, dim=1)
                        norm_a = norm_a.mean()
                        self.adjust_step_critic(norm_a)

                    a_prime = a_prime * (1 - self.alpha3) + adv_prime * self.alpha3
                    log_pi_a_prime = log_pi_a_prime * (1 - self.alpha3)
                    target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
                    target_Q = torch.min(target_Q1, target_Q2)
                else:
                    # self.alpha3 = random.uniform(0.03,0.18)
                    a_prime, log_pi_a_prime = self.actor(s_prime, False, True)
                    adv_prime, _ = self.adversary(s_prime, False, False)
                    a_prime = a_prime * (1 - self.alpha3) + adv_prime * self.alpha3*self.sigm

                    log_pi_a_prime = log_pi_a_prime * (1 - self.alpha3)
                    target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
                    target_Q = torch.min(target_Q1, target_Q2)
            else:
                a_prime, log_pi_a_prime = self.actor(s_prime, False, True)
                adv_prime, _ = self.adversary(s_prime, False, False)
                Q1, Q2 = self.q_critic_target(s_prime, a_prime)
                adv_Q1, adv_Q2 = self.q_critic_target(s_prime, adv_prime)
                target_q = torch.min(Q1, Q2)
                target_adv_q = torch.min(adv_Q1, adv_Q2)
                target_Q = target_q * (1 - self.alpha1) + target_adv_q * self.alpha1
            target_Q = r + (1 - dead_mask) * self.gamma * (target_Q - self.alpha * log_pi_a_prime)
            current_Q1, current_Q2 = self.q_critic(s, a)
            value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.q_critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            # for name,parms in self.q_critic.named_parameters():
            #     print('-->name:',name,'-->grad_requirs:',parms.requires_grad,'-->grad_value:',parms.grad)
            self.q_critic_optimizer.step()
            value_loss = value_loss.item()
        else:
            value_loss = 0

        if adversary_update:
            # Train adversary
            self.adversary_optimizer.zero_grad()
            if mdp_type == 'nr_mdp':
                if adaptive_step:
                    with torch.no_grad():
                        real_a, _ = self.actor(s, False, False)
                    adv_a, _ = self.adversary(s, False, False)
                    action = (1 - self.alpha3) * real_a + self.alpha3 * adv_a
                    # action = (1-self.alpha3)*real_a + self.alpha3 * adv_grad_2
                    current_Q1, current_Q2 = self.q_critic(s, action)
                    Q = torch.min(current_Q1, current_Q2)
                else:
                    # self.alpha1 = random.uniform(0.03,0.18)
                    with torch.no_grad():
                        real_a, _ = self.actor(s, False, False)
                    adv_a, _ = self.adversary(s, False, False)
                    action = (1 - self.alpha3) * real_a + self.alpha3 * adv_a*self.sigm

                    current_Q1, current_Q2 = self.q_critic(s, action)
                    Q = torch.min(current_Q1, current_Q2)
            else:
                adv, _ = self.adversary(s, False, False)
                current_Q1, current_Q2 = self.q_critic(s, adv)
                Q = torch.min(current_Q1, current_Q2) * self.alpha1
            adversary_loss = Q.mean()
            adversary_loss.backward()
            self.adversary_optimizer.step()
            adversary_loss = adversary_loss.item()
            policy_loss = 0
        else:

            if robust_update == 'full':
                # Train actor
                self.actor_optimizer.zero_grad()
                if mdp_type == 'nr_mdp':
                    if adaptive_step:
                        with torch.no_grad():
                            adv_a, _ = self.adversary(s, False, False)
                        a, log_pi_a = self.actor(s, False, True)
                        action = (1 - self.alpha3) * a + self.alpha3 * adv_a
                        current_Q1, current_Q2 = self.q_critic(s, action)
                        Q = torch.min(current_Q1, current_Q2)
                        policy_loss = ((1 - self.alpha3) * self.alpha * log_pi_a - Q).mean()
                    else:
                        # self.alpha3 = random.uniform(0.03,0.18)
                        with torch.no_grad():
                            adv_a, _ = self.adversary(s, False, False)
                        a, log_pi_a = self.actor(s, False, True)
                        action = (1 - self.alpha3) * a + adv_a * self.alpha3*self.sigm

                        current_Q1, current_Q2 = self.q_critic(s, action)
                        Q = torch.min(current_Q1, current_Q2)
                        policy_loss = ((1 - self.alpha3) * self.alpha * log_pi_a - Q).mean()
                else:
                    action, log_pi_a = self.actor(s, False, True)
                    current_Q1, current_Q2 = self.q_critic(s, action)
                    Q = torch.min(current_Q1, current_Q2) * (1 - self.alpha1)
                    policy_loss = (self.alpha * log_pi_a - Q).mean()

                policy_loss.backward()
                self.actor_optimizer.step()
                policy_loss = policy_loss.item()
                if adaptive_step:
                    adv_adjust_step = self.alpha3.item()

                adversary_loss = 0
                self.update_alpha(log_pi_a)
            else:
                policy_loss = 0
                adversary_loss = 0

        return value_loss, policy_loss, adversary_loss, adv_adjust_step, gradient_norm

    def update_non_robust(self, replay_buffer):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)
        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime)  # log_pi_a_prime是策略的熵？
            target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (
                        target_Q - self.alpha * log_pi_a_prime)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        value_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        policy_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        for params in self.q_critic.parameters():
            params.requires_grad = True
        policy_loss = policy_loss.item()
        adversary_loss = 0
        self.update_alpha(log_pi_a)
        return value_loss.item(), policy_loss, adversary_loss

    def update_alpha(self, log_pi_a):
        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
    # adaptive perturbation
    def adjust_step(self, a1):
        b = 0.5
        self.alpha1 = (self.alpha1 - 0.01 * torch.sign(a1 - self.a1) * torch.sigmoid(torch.abs(a1 - self.a1))).clamp(
            0.03, 0.2)
        # self.alpha1 = self.alpha1 * 0.1
        self.a1 = (1 - b) * self.a1 + b * a1

    def adjust_step_sample(self, a2):
        b = 0.5
        self.alpha2 = (self.alpha2 - 0.01 * torch.sign(a2 - self.a2) * torch.sigmoid(torch.abs(a2 - self.a2))).clamp(
            0.03, 0.2)
        # self.alpha2 = self.alpha2 * 0.1
        self.a2 = (1 - b) * self.a2 + b * a2

    def adjust_step_critic(self, a3):
        b = 0.5
        self.alpha3 = (self.alpha3 - 0.01 * torch.sign(a3 - self.a3) * torch.sigmoid(torch.abs(a3 - self.a3))).clamp(
            0.03, 0.2)
        # self.alpha3 = self.alpha3 * 0.1
        self.a3 = (1 - b) * self.a3 + b * a3

    def adjust_step_adv(self, a4):
        b = 0.5
        self.alpha4 = (self.alpha4 - 0.01 * torch.sign(a4 - self.a4) * torch.sigmoid(torch.abs(a4 - self.a4))).clamp(
            0.03, 0.2)
        self.a4 = (1 - b) * self.a4 + b * a4

    def train(self, replay_buffer, mdp_type='mdp', adversary_update=False, adaptive_step=False):
        if mdp_type != 'mdp':
            robust_update_type = 'full'
        else:
            robust_update_type = None
        value_loss = 0
        policy_loss = 0
        adversary_loss = 0
        adv_adjust_step = 0
        gradient_norms = 0
        if robust_update_type is not None:
            _value_loss, _policy_loss, _adversary_loss, _adv_adjust_step, _gradient_norm = self.update_robust(
                replay_buffer, adversary_update, mdp_type, robust_update_type, adaptive_step)
            value_loss += _value_loss
            policy_loss += _policy_loss
            adversary_loss += _adversary_loss
            adv_adjust_step += _adv_adjust_step
            gradient_norms += _gradient_norm
        else:
            _value_loss, _policy_loss, _adversary_loss = self.update_non_robust(replay_buffer)
            value_loss += _value_loss
            policy_loss += _policy_loss
            adversary_loss += _adversary_loss

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return value_loss, policy_loss, adversary_loss, adv_adjust_step, gradient_norms

    def save(self, base_dir, episode):
        torch.save(self.actor.state_dict(), base_dir + '/actor{}.pth'.format(episode))
        torch.save(self.adversary.state_dict(), base_dir + "/adversary{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), base_dir + "/q_critic{}.pth".format(episode))

    def load(self, base_dir, episode):
        self.actor.load_state_dict(torch.load(base_dir + "/actor{}.pth".format(episode), map_location='cuda:0'))
        self.adversary.load_state_dict(torch.load(base_dir + "/adversary{}.pth".format(episode), map_location='cuda:0'))
        self.q_critic.load_state_dict(torch.load(base_dir + "/q_critic{}.pth".format(episode), map_location='cuda:0'))

    def load_nr(self, base_dir):
        self.actor.load_state_dict(torch.load(base_dir + "/actor4.pth", map_location='cuda:0'))
        self.adversary.load_state_dict(torch.load(base_dir + "/adversary4.pth", map_location='cuda:0'))
        self.q_critic.load_state_dict(torch.load(base_dir + "/q_critic4.pth", map_location='cuda:0'))

    def eval(self):
        self.actor.eval()
        self.adversary.eval()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev

        """Apply parameter noise to adversary model, for exploration"""
        hard_update(self.adversary_perturbed, self.adversary)
        params = self.adversary_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev
