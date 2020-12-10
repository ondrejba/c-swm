import copy as cp
import utils

import numpy as np

import torch
from torch import nn



class ContrastiveSWM(nn.Module):
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

        encoder_input_dim = np.prod(width_height)
        if self.rot:
            encoder_input_dim = 7 #Hard-coded for now

        self.obj_encoder = mlp_class(input_dim=encoder_input_dim,
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
        obj = self.obj_extractor(obs)
        return self.obj_encoder(obj)


class C4Conv(nn.Module):
    """C_4 Convolution"""
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.rand(4, size_out, size_in)
        weights -= 0.5
        k = 1 / torch.sqrt(torch.tensor(size_in, dtype=torch.float))
        weights *= k
        self.weights = torch.nn.parameter.Parameter(weights)
        bias = torch.rand(size_out)
        bias -= 0.5
        bias *= k
        self.bias = torch.nn.parameter.Parameter(bias)
        mat = torch.stack([torch.roll(self.weights,i,dims=0) for i in range(4)],dim=0)
        self.register_buffer('mat', mat)
    
    def updateKernel(self):
        self.mat=torch.stack([torch.roll(self.weights,i,dims=0) for i in range(4)],dim=0)

    def forward(self, x):
        #TODO: really should only call after update
        self.updateKernel()
        w_times_x= torch.einsum('ghij,...hj->...gi',self.mat, x)
        return torch.add(w_times_x, self.bias)  # w times x + b


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

        self.fc1 = C4Conv(self.input_dim, hidden_dim)
        self.fc2 = C4Conv(hidden_dim, hidden_dim)
        self.fc3 = C4Conv(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        """ 
        input: 2D Shapes (batch, num_objects, 5, 5)

        """
        h_flat = self.orbit_stack(ins)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)

    def rot90(self, x, exp = 1):
        if (exp == 0):
            return x
        else:
            return self.rot90(x.flip(3).permute(0,1,3,2), exp - 1)

    def orbit_stack(self, x):
        """
        Input: (batch, num_objects, H, H)
        H should be odd and input is square image
        """
        H = x.size(-1)
        c = (H - 1) // 2
        n_orbits = 1 + c + (c)**2
        
        out = torch.zeros((x.size(0),x.size(1),n_orbits,4),device=self.fc1.weights.device)

        out[:,:,0,:] = ((x[:,:,c,c]).unsqueeze(2)).expand(-1,-1,4)
        for i in range(4):
            out[:,:,1:,i] = self.rot90(x,i)[:,:,:c,:c+1].reshape(x.size(0),x.size(1),-1)
        
        return out.permute(0,1,3,2)


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
            C4Conv(self.input_dim*2, hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, hidden_dim))
        
        node_input_dim = hidden_dim + self.input_dim + 1 #action_dim = 1 vs. 4

        self.node_mlp = nn.Sequential(
            C4Conv(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            C4Conv(hidden_dim, self.input_dim))

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
        out = torch.cat([source, target], dim=2)

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
            return ret  #Right?
        else:
            return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=2)
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
        #print("states:",states.shape)
        #node_attr = states.view(-1, self.input_dim)
        node_attr = states.reshape(-1, 4, self.input_dim)
        #print("node_shape:",node_attr.shape)

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
            node_attr = torch.cat([node_attr, torch.flip(action_vec.unsqueeze(2),dims=[1])], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, 4, -1)

        if self.immovable_bit:
            # object embeddings have an additional bit for movable/immovable objects
            # we do not need to predict that
            node_attr = node_attr[:, :, :self.input_dim - 1]

        return node_attr


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))
    
    
class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class SplitEncoderMLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(SplitEncoderMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects

        # for now, I hard-coded the immovable objects setup
        assert num_objects == 5

        self.e_immovable = EncoderMLP(input_dim, output_dim, hidden_dim, 2, act_fn=act_fn)
        self.e_movable = EncoderMLP(input_dim, output_dim, hidden_dim, 3, act_fn=act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h_immovable = self.e_immovable(h_flat[:, :2, :])
        h_movable = self.e_movable(h_flat[:, 2:, :])
        return torch.cat([h_immovable, h_movable], dim=1)


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = utils.to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1],
                      self.output_size[2])


class DecoderCNNSmall(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNSmall, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=10, stride=10)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNLarge, self).__init__()

        width, height = output_size[1], output_size[2]

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=3, padding=1)

        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.ln2 = nn.BatchNorm2d(hidden_dim)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)
        self.act4 = utils.get_act_fn(act_fn)
        self.act5 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        h = self.act4(self.ln1(self.deconv2(h)))
        h = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)
