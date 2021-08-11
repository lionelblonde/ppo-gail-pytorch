from collections import defaultdict
import os
import os.path as osp

from gym import spaces

import numpy as np
import torch
import torch.nn.utils as U
from torch.utils.data import DataLoader

from helpers import logger
from helpers.dataset import Dataset
from helpers.console_util import log_env_info, log_module_info
from helpers.math_util import LRScheduler
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.nets import GaussPolicy, Value, CatPolicy
from agents.gae import gae

from agents.rnd import RandomNetworkDistillation


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class PPOAgent(object):

    def __init__(self, env, device, hps):
        self.env = env
        self.device = device
        self.hps = hps

        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        assert len(self.ob_shape) in [1, 3], "invalid observation space shape."
        self.hps.visual = len(self.ob_shape) == 3  # add it to hps, passed on to the nets
        self.is_discrete = isinstance(self.ac_space, spaces.Discrete)
        self.ac_dim = self.ac_space.n if self.is_discrete else self.ac_shape[-1]

        if self.hps.clip_norm <= 0:
            logger.info("clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Create observation normalizer that maintains running statistics
        if not self.hps.visual:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)
        else:
            self.rms_obs = None

        # Create nets
        Policy = CatPolicy if self.is_discrete else GaussPolicy
        self.policy = Policy(self.env, self.hps, self.rms_obs).to(self.device)
        sync_with_root(self.policy)
        if not self.hps.shared_value:
            self.value = Value(self.env, self.hps, self.rms_obs).to(self.device)
            sync_with_root(self.value)

        # Set up the optimizer
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hps.p_lr)
        if not self.hps.shared_value:
            self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.hps.v_lr)

        # Set up lr scheduler
        self.scheduler = LRScheduler(
            optimizer=self.p_optimizer,
            initial_lr=self.hps.p_lr,
            lr_schedule=self.hps.lr_schedule,
            total_num_steps=self.hps.num_timesteps,
        )

        log_module_info(logger, 'policy', self.policy)
        if not self.hps.shared_value:
            log_module_info(logger, 'value', self.policy)

        if self.hps.rnd_explo:
            self.rnd = RandomNetworkDistillation(
                self.env,
                self.device,
                self.hps,
                self.rms_obs,
            )

    def predict(self, ob, sample_or_mode):
        # Create tensor from the state (`require_grad=False` by default)

        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict an action
        ac = self.policy.sample(ob) if sample_or_mode else self.policy.mode(ob)
        # Also retrieve the log-probability associated with the picked action
        logp = self.policy.logp(ob, ac)
        # Place on cpu and collapse into one dimension
        if self.hps.shared_value:
            v = self.policy.value(ob).cpu().detach().numpy().flatten()
        else:
            v = self.value(ob).cpu().detach().numpy().flatten()
        ac = ac.cpu().detach().numpy().flatten()
        logp = logp.cpu().detach().numpy().flatten()
        return ac, v, logp

    def update_policy_value(self, rollout, iters_so_far):
        """Train the agent"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Augment `rollout` with GAE (Generalized Advantage Estimation), which among
        # other things adds the GAE estimate of the MC estimate of the return
        gae(rollout, self.hps.gamma, self.hps.gae_lambda, rew_key='env_rews')

        # Standardize advantage function estimate
        rollout['advs'] = ((rollout['advs'] - rollout['advs'].mean()) /
                           (rollout['advs'].std() + 1e-8))

        # Create DataLoader object to iterate over transitions in rollouts
        keys = ['obs0', 'acs', 'logps', 'vs', 'advs', 'td_lam_rets']
        dataset = Dataset({k: rollout[k] for k in keys})
        dataloader = DataLoader(
            dataset,
            self.hps.batch_size,
            shuffle=True,
            drop_last=False,  # no compatibility issue, only used for policy alone
        )

        for _ in range(self.hps.optim_epochs_per_iter):

            for chunk in dataloader:

                # Transfer to device
                state = chunk['obs0'].to(self.device)
                action = chunk['acs'].to(self.device)
                logp_old = chunk['logps'].to(self.device)
                v_old = chunk['vs'].to(self.device)
                advantage = chunk['advs'].to(self.device)
                td_lam_return = chunk['td_lam_rets'].to(self.device)

                if not self.hps.visual:
                    # Update the observation normalizer
                    self.rms_obs.update(state)

                # Policy loss
                entropy_loss = -self.hps.p_ent_reg_scale * self.policy.entropy(state).mean()
                logp = self.policy.logp(state, action)
                ratio = torch.exp(logp - logp_old)
                surrogate_loss_a = -advantage * ratio
                surrogate_loss_b = -advantage * ratio.clamp(1.0 - self.hps.eps,
                                                            1.0 + self.hps.eps)
                clip_loss = torch.max(surrogate_loss_a, surrogate_loss_b).mean()
                kl_approx = (logp - logp_old).mean()
                clip_frac = (ratio - 1.0).abs().gt(self.hps.eps).float().mean()
                # Value loss
                if self.hps.shared_value:
                    v = self.policy.value(state)
                else:
                    v = self.value(state)
                clip_v = v_old + (v - v_old).clamp(-self.hps.eps, self.hps.eps)
                v_loss_a = (clip_v - td_lam_return).pow(2)
                v_loss_b = (v - td_lam_return).pow(2)
                v_loss = torch.max(v_loss_a, v_loss_b).mean()
                if self.hps.shared_value:
                    p_loss = clip_loss + entropy_loss + (self.hps.baseline_scale * v_loss)
                else:
                    p_loss = clip_loss + entropy_loss

                # Log metrics
                metrics['entropy_loss'].append(entropy_loss)
                metrics['clip_loss'].append(clip_loss)
                metrics['kl_approx'].append(kl_approx)
                metrics['clip_frac'].append(clip_frac)
                metrics['v_loss'].append(v_loss)
                metrics['p_loss'].append(p_loss)

                # # Early-stopping, based on KL value
                # kl_approx_mpi = mpi_mean_like(kl_approx.detach().cpu().numpy())  # none or all
                # kl_thres = 0.05  # not (yet) hyperparameterized
                # if iters_so_far > 20 and kl_approx_mpi > 1.5 * kl_thres:
                #     if DEBUG:
                #         logger.info("triggered early-stopping")
                #     # Skip gradient update
                #     break

                # Update parameters
                self.p_optimizer.zero_grad()
                p_loss.backward()
                average_gradients(self.policy, self.device)
                if self.hps.clip_norm > 0:
                    U.clip_grad_norm_(self.policy.parameters(), self.hps.clip_norm)
                self.p_optimizer.step()

                _lr = self.scheduler.step(steps_so_far=iters_so_far * self.hps.rollout_len)
                if DEBUG:
                    logger.info(f"lr is {_lr} after {iters_so_far} timesteps")

                if not self.hps.shared_value:
                    self.v_optimizer.zero_grad()
                    v_loss.backward()
                    average_gradients(self.value, self.device)
                    self.v_optimizer.step()

            if self.hps.rnd_explo:
                # In accordance with RND's pseudo-code, train as many times as the policy
                self.rnd.update(dataloader)  # ignore returned var

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics, _lr

    def save(self, path, iters_so_far):
        if self.hps.obs_norm:
            torch.save(self.rms_obs.state_dict(), osp.join(path, f"rms_obs_{iters_so_far}.pth"))
        torch.save(self.policy.state_dict(), osp.join(path, f"policy_{iters_so_far}.pth"))
        if not self.hps.shared_value:
            torch.save(self.value.state_dict(), osp.join(path, f"value_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        if self.hps.obs_norm:
            self.rms_obs.load_state_dict(torch.load(osp.join(path, f"rms_obs_{iters_so_far}.pth")))
        self.policy.load_state_dict(torch.load(osp.join(path, f"policy_{iters_so_far}.pth")))
        if not self.hps.shared_value:
            self.value.load_state_dict(torch.load(osp.join(path, f"value_{iters_so_far}.pth")))
