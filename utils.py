"""Utility functions."""

import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        return obs, action, next_obs


class StateTransitionsDatasetStateIds(StateTransitionsDataset):

    def __init__(self, hdf5_file):

        super(StateTransitionsDatasetStateIds, self).__init__(hdf5_file)

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        state_ids = self.experience_buffer[ep]['state_ids'][step]
        next_state_ids = self.experience_buffer[ep]['next_state_ids'][step]

        return obs, action, next_obs, state_ids, next_state_ids


class StateTransitionsDatasetStateIdsNegs(StateTransitionsDataset):

    def __init__(self, hdf5_file):

        super(StateTransitionsDatasetStateIdsNegs, self).__init__(hdf5_file)

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        state_ids = self.experience_buffer[ep]['state_ids'][step]
        next_state_ids = self.experience_buffer[ep]['next_state_ids'][step]

        in_out = np.random.choice([True, False])

        if in_out:
            # choose from inside the episode
            random_step = np.random.randint(len(self.experience_buffer[ep]['obs']))
            neg_obs = to_float(self.experience_buffer[ep]['obs'][random_step])
            neg_state_id = self.experience_buffer[ep]['state_ids'][random_step]
        else:
            # choose from outside the episode
            random_ep = np.random.randint(len(self.experience_buffer))
            neg_obs = to_float(self.experience_buffer[random_ep]['obs'][step])
            neg_state_id = self.experience_buffer[random_ep]['state_ids'][step]

        return obs, action, next_obs, state_ids, next_state_ids, neg_obs, neg_state_id


class StateTransitionsDatasetTwins(StateTransitionsDataset):

    MODE_RANDOM = 0
    MODE_NEXT = 1
    MODE_WINDOW = 2

    def __init__(self, hdf5_file, mode, window_size=5):

        super(StateTransitionsDatasetTwins, self).__init__(hdf5_file)

        assert mode in [self.MODE_RANDOM, self.MODE_NEXT, self.MODE_WINDOW]

        self.mode = mode
        self.window_size = window_size

    def __getitem__(self, idx):

        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        state_ids = self.experience_buffer[ep]['state_ids'][step]

        if self.mode == self.MODE_RANDOM:
            # random step from the same episode
            twin_step = np.random.randint(len(self.experience_buffer[ep]['obs']))
        elif self.mode == self.MODE_WINDOW:
            # random step within a window of the current step
            twin_step = np.random.randint(
                max(0, step - self.window_size), min(len(self.experience_buffer[ep]['obs']), step + self.window_size)
            )
        else:
            # next step or the previous step if we are at the end
            if step == len(self.experience_buffer[ep]['obs']) - 1:
                twin_step = step - 1
            else:
                twin_step = step + 1

        twin_obs = to_float(self.experience_buffer[ep]['obs'][twin_step])
        twin_action = self.experience_buffer[ep]['action'][twin_step]
        twin_next_obs = to_float(self.experience_buffer[ep]['next_obs'][twin_step])
        twin_state_ids = self.experience_buffer[ep]['state_ids'][twin_step]

        return obs, action, next_obs, state_ids, twin_obs, twin_action, twin_next_obs, twin_state_ids


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action'][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions


class PathDatasetStateIds(PathDataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        super(PathDatasetStateIds, self).__init__(hdf5_file, path_length)

    def __getitem__(self, idx):
        observations = []
        actions = []
        state_ids = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action'][i]
            state_id = self.experience_buffer[idx]['state_ids'][i]
            observations.append(obs)
            actions.append(action)
            state_ids.append(state_id)

        obs = to_float(self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        state_id = self.experience_buffer[idx]['next_state_ids'][self.path_length - 1]

        observations.append(obs)
        state_ids.append(state_id)

        return observations, actions, state_ids

def css_to_ssc(image):
    return image.transpose((1, 2, 0))

def to_np(x):
    return x.detach().cpu().numpy()
