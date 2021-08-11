import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import logger
from helpers.console_util import log_module_info
from agents.nets import init
from helpers.distributed_util import RunMoms


STANDARDIZED_OB_CLAMPS = [-5., 5.]


class PredNet(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(PredNet, self).__init__()
        self.hps = hps
        assert not self.hps.visual, "network not adapted to visual input (for now)"
        ob_dim = env.observation_space.shape[0]
        if self.hps.wrap_absorb:
            ob_dim += 1
        self.leak = 0.1
        if self.hps.rnd_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_3', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

    def forward(self, obs):
        if self.hps.rnd_batch_norm:
            # Apply normalization
            if self.hps.wrap_absorb:
                obs_ = obs.clone()[:, 0:-1]
                obs_ = self.rms_obs.standardize(obs_).clamp(*STANDARDIZED_OB_CLAMPS)
                obs = torch.cat([obs_, obs[:, -1].unsqueeze(-1)], dim=-1)
            else:
                obs = self.rms_obs.standardize(obs).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            obs = obs.clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(obs)
        return x


class TargNet(PredNet):

    def __init__(self, env, hps, rms_obs):
        super(TargNet, self).__init__(env, hps, rms_obs)
        ob_dim = env.observation_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
        self.hps = hps
        self.leak = 0.1
        if self.hps.rnd_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_3', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

        # Prevent the weights from ever being updated
        for param in self.fc_stack.parameters():
            param.requires_grad = False


class RandomNetworkDistillation(object):

    """Difference compared to original RND:
    we add the intrinsic reward to the main reward and learn a single advantage,
    as opposed to learning one advantage for each, since we do not play with different horizons.
    """

    def __init__(self, env, device, hps, rms_obs):
        self.env = env
        self.device = device
        self.hps = hps
        self.rms_obs = rms_obs

        # Create nets
        self.pred_net = PredNet(self.env, self.hps, self.rms_obs).to(self.device)
        self.targ_net = TargNet(self.env, self.hps, self.rms_obs).to(self.device)  # fixed, not trained

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=self.hps.rnd_lr)

        # Define reward normalizer
        self.rms_int_rews = RunMoms(shape=(1,), use_mpi=False)

        log_module_info(logger, 'RND Pred Network', self.pred_net)
        log_module_info(logger, 'RND Targ Network', self.targ_net)

    def update(self, dataloader):
        """Update the RND predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        for batch in dataloader:
            logger.info("updating rnd predictor")

            # Transfer to device
            state = batch['obs0'].to(self.device)

            # Compute loss
            _loss = F.mse_loss(
                self.pred_net(state),
                self.targ_net(state),
                reduction='none',
            )
            loss = _loss.mean(dim=-1)
            mask = loss.clone().detach().data.uniform_().to(self.device)
            mask = (mask < self.hps.proportion_of_exp_per_rnd_update).float()
            loss = (mask * loss).sum() / torch.max(torch.Tensor([1.]), mask.sum())
            metrics['loss'].append(loss)

            # Update running moments for intrinsic rewards
            int_rews = F.mse_loss(
                self.pred_net(state),
                self.targ_net(state),
                reduction='none',
            ).mean(dim=-1, keepdim=True).detach()
            self.rms_int_rews.update(int_rews)

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def get_int_rew(self, next_state):  # name purposefully explicit
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.Tensor(next_state)
        # Transfer to cpu
        next_state = next_state.cpu()
        # Compute intrinsic reward
        int_rew = F.mse_loss(
            self.pred_net(next_state),
            self.targ_net(next_state),
            reduction='none',
        ).mean(dim=-1, keepdim=True).detach()
        # Normalize intrinsic reward
        int_rew = self.rms_int_rews.divide_by_std(int_rew)
        return int_rew
