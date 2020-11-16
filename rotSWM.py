import copy as cp
import utils

import numpy as np

import torch
from torch import nn

from modules import EncoderCNNSmall, EncoderCNNMedium, EncoderCNNLarge, TransitionGNN, DecoderMLP, DecoderCNNSmall, DecoderCNNMedium, DecoderCNNMedium


class RotContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
        same_ep_neg: Sample negative samples from the same episode.
    """
    def __init__(self,
                 embedding_dim,
                 input_dims,
                 hidden_dim,
                 action_dim,
                 num_objects,
                 hinge=1.,
                 sigma=0.5,
                 encoder='large',
                 ignore_action=False,
                 copy_action=False,
                 split_mlp=False,
                 same_ep_neg=False,
                 only_same_ep_neg=False,
                 immovable_bit=False,
                 split_gnn=False):
        super(ContrastiveSWM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.split_mlp = split_mlp
        self.same_ep_neg = same_ep_neg
        self.only_same_ep_neg = only_same_ep_neg
        self.split_gnn = split_gnn

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(input_dim=num_channels,
                                                 hidden_dim=hidden_dim // 16,
                                                 num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(input_dim=num_channels,
                                                  hidden_dim=hidden_dim // 16,
                                                  num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(input_dim=num_channels,
                                                 hidden_dim=hidden_dim // 16,
                                                 num_objects=num_objects)

        mlp_class = EncoderMLP
        if self.split_mlp:
            mlp_class = SplitEncoderMLP

        self.obj_encoder = mlp_class(input_dim=np.prod(width_height),
                                     hidden_dim=hidden_dim,
                                     output_dim=embedding_dim,
                                     num_objects=num_objects)

        self.transition_model = TransitionGNN(input_dim=embedding_dim,
                                              hidden_dim=hidden_dim,
                                              action_dim=action_dim,
                                              num_objects=num_objects,
                                              ignore_action=ignore_action,
                                              copy_action=copy_action,
                                              immovable_bit=immovable_bit,
                                              split_gnn=split_gnn)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):

        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)

        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge -
            self.energy(state, action, neg_state, no_trans=True)).mean()

        if self.same_ep_neg:
            ep_size = state.size(1)
            neg_state = state.clone()

            # same perm for all batches
            #perm = np.random.permutation(np.arange(ep_size))
            #neg_state[:, :] = neg_state[:, perm]

            # different perm for each batch
            perm = np.stack([
                np.random.permutation(np.arange(ep_size))
                for _ in range(batch_size)
            ],
                            axis=0)
            indices = np.arange(batch_size)[:, np.newaxis].repeat(ep_size,
                                                                  axis=1)
            neg_state[:, :] = neg_state[indices, perm]

            if self.only_same_ep_neg:
                # overwrite the negative loss calculated above
                self.neg_loss = torch.max(
                    zeros, self.hinge - self.energy(
                        state, action, neg_state, no_trans=True)).mean()
            else:
                # average negative loss over in-episode and out-of-episode samples
                self.neg_loss += torch.max(
                    zeros, self.hinge - self.energy(
                        state, action, neg_state, no_trans=True)).mean()
                self.neg_loss /= 2

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class RotEncoderMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 num_objects,
                 act_fn='relu'):

        super().__init__()
        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        """ 
        input: 2D Shapes (batch, num_objects, 5, 5)

        """
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)

    def rot90(self, x, exp = 1):
        if (exp == 0):
            return x
        else:
            return rot90(x.flip(3).permute(0,1,3,2), exp - 1)

    def orbit_stack(self, x):
        """
        Input: (batch, num_objects, H, H)
        H should be odd and input is square image
        """
        H = x.size(-1)
        c = (H - 1) // 2
        n_orbits = 1 + c + (c)**2
        
        out = torch.zeros((x.size(0),x.size(1),n_orbits,4))
        
        out[:,:,0,:] = ((x[:,:,c,c]).unsqueeze(2)).expand(-1,-1,4)
        for i in range(4):
            out[:,:,1:,i] = rot90(x,i)[:,:,:c,:c+1].reshape(x.size(0),x.size(1),-1)
        
        return out

