import copy as cp
import utils

import numpy as np

import torch
from torch import nn

from modules import EncoderCNNSmall, EncoderCNNMedium, EncoderCNNLarge, DecoderMLP, DecoderCNNSmall, DecoderCNNMedium, DecoderCNNMedium


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
                 split_gnn=False,
                 rot=False):
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
        self.rot = rot

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
            mlp_class = SplitEncoderMLP    #To do, make switch  
        if self.rot:
            mlp_class = RotEncoderMLP

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
                                              split_gnn=split_gnn,
                                              rot=rot)

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

class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 ignore_action=False, copy_action=False, act_fn='relu',
                 immovable_bit=False, split_gnn=False, rot=False):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.immovable_bit = immovable_bit
        self.split_gnn = split_gnn
        self.rot = rot

        if self.immovable_bit:
            self.input_dim += 1

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_dim*2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + self.input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, self.input_dim))

        if self.split_gnn:
            self.edge_mlp1 = self.edge_mlp
            self.edge_mlp2 = cp.deepcopy(self.edge_mlp1)
            self.edge_mlp3 = cp.deepcopy(self.edge_mlp1)

            self.node_mlp1 = self.node_mlp
            self.node_mlp2 = cp.deepcopy(self.node_mlp1)

            del self.node_mlp
            del self.edge_mlp

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr, source_indices=None, target_indices=None):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)

        if self.split_gnn:
            ret = torch.zeros((out.size(0), self.hidden_dim), dtype=out.dtype, device=out.device)
            mask1 = np.logical_and(np.logical_or(source_indices == 0, source_indices == 1),
                                   np.logical_or(target_indices == 0, target_indices == 1))
            mask2 = np.logical_and(np.logical_or(np.logical_or(source_indices == 2, source_indices == 3), source_indices == 4),
                                   np.logical_or(np.logical_or(target_indices == 2, target_indices == 3), target_indices == 4))
            mask3 = np.logical_and(np.logical_not(mask1), np.logical_not(mask2))

            ret[mask1] = self.edge_mlp1(out[mask1])
            ret[mask2] = self.edge_mlp2(out[mask2])
            ret[mask3] = self.edge_mlp3(out[mask3])
        else:
            return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr

        if self.split_gnn:
            ret = torch.zeros((out.size(0), self.input_dim), dtype=out.dtype, device=out.device)
            obj12_indices = np.concatenate(
                [np.arange(0, out.size(0), self.num_objects),
                 np.arange(1, out.size(0), self.num_objects)])
            obj345_indices = np.concatenate(
                [np.arange(2, out.size(0), self.num_objects),
                 np.arange(3, out.size(0), self.num_objects),
                 np.arange(4, out.size(0), self.num_objects)])
            ret[obj12_indices] = self.node_mlp1(out[obj12_indices])
            ret[obj345_indices] = self.node_mlp2(out[obj345_indices])
            return ret
        else:
            return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        if self.immovable_bit:
            # add movable/immovable bit (the first two objects are immovable, this is hardcoded for now)
            tmp = torch.zeros_like(states[:, :, 0:1])
            tmp[:, :2, :] = 1.0
            states = torch.cat([states, tmp], dim=2)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr, source_indices=row % self.num_objects,
                target_indices=col % self.num_objects)

        if not self.ignore_action:

            if self.copy_action:
                action_vec = utils.to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = utils.to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        if self.immovable_bit:
            # object embeddings have an additional bit for movable/immovable objects
            # we do not need to predict that
            node_attr = node_attr[:, :, :self.input_dim - 1]

        return node_attr