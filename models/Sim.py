import torch
import torch.nn as nn
from constants import Constants


class Sim(nn.Module):

    def __init__(self, encoder, target_encoder, config):

        super(Sim, self).__init__()

        c = config
        self.gamma = c[Constants.GAMMA]

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.sync()

    def loss(self, states, actions, next_states, beta, state_ids, metric):

        # we have N transitions, create all N^2 possible transition pairs
        state_pairs = self.all_image_pairs_(states)
        next_state_pairs = self.all_image_pairs_(next_states)
        action_pairs = self.all_pairs_(actions[:, None])
        if len(state_ids.shape) == 1:
            state_id_pairs = self.all_pairs_(state_ids[:, None])
        else:
            state_id_pairs = self.all_pairs_(state_ids)

        # only select pairs where a_1 == a_2
        # TODO: maybe don't do pairs of the same state?
        mask = action_pairs[:, 0] == action_pairs[:, 1]
        state_pairs = state_pairs[mask]
        next_state_pairs = next_state_pairs[mask]
        state_id_pairs = state_id_pairs[mask]

        # get reward distances
        if state_id_pairs.shape[1] > 2:
            size = state_id_pairs.shape[1] // 2
            reward_dists = 1 - (torch.all(state_id_pairs[:, :size] == state_id_pairs[:, size:], dim=1)).float()
        else:
            reward_dists = (state_id_pairs[:, 0] != state_id_pairs[:, 1]).float()

        # get state pair distances and next state pair distances using online and target networks
        state_dists = self.encoder(state_pairs)[:, 0]
        state_target_dists = self.target_encoder(state_pairs).detach()[:, 0]
        next_state_target_dists = self.target_encoder(next_state_pairs).detach()[:, 0]

        # calculate targets and compute squared loss
        target1 = (1 - self.gamma) * reward_dists + self.gamma * beta * next_state_target_dists
        target2 = beta * state_target_dists
        target = torch.max(target1, target2)

        loss = (state_dists - target) ** 2
        loss = loss.mean()

        if state_ids is not None and metric is not None:
            return loss, self.abs_error(state_dists, state_id_pairs, metric)
        else:
            return loss

    def abs_error(self, state_dists, state_id_pairs, metric):

        gt_dists = metric[state_id_pairs[:, 0], state_id_pairs[:, 1]]
        return torch.mean(torch.abs(state_dists - gt_dists))

    def all_image_pairs_(self, states):

        batch_size = states.size(0)
        latent_size = states.size(1)
        height = states.size(2)
        width = states.size(3)

        states1 = states[:, None, :].repeat(1, batch_size, 1, 1, 1)
        states2 = states[None, :, :].repeat(batch_size, 1, 1, 1, 1)

        return torch.cat([states1, states2], dim=2).reshape((batch_size ** 2, 2 * latent_size, height, width))

    def all_pairs_(self, states):

        batch_size = states.size(0)
        latent_size = states.size(1)

        states1 = states[:, None, :].repeat(1, batch_size, 1)
        states2 = states[None, :, :].repeat(batch_size, 1, 1)

        return torch.cat([states1, states2], dim=2).reshape((batch_size ** 2, 2 * latent_size))

    def sync(self):

        self.target_encoder.load_state_dict(self.encoder.state_dict())
