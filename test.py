import torch
from torch import nn
from modules2.ConvEncoder import ConvEncoder
from modules2.FCEncoder import FCEncoder
from models.Sim import Sim
from logger import Logger
from constants import Constants


def make_pairwise_encoder():

    logger = Logger()

    cnn_config = {
        Constants.INPUT_SIZE: (50, 50, 6),
        Constants.FILTER_SIZES: [5, 5, 5],
        Constants.FILTER_COUNTS: [16, 32, 64],
        Constants.STRIDES: [2, 2, 2],
        Constants.USE_BATCH_NORM: False,
        Constants.ACTIVATION_LAST: True,
        Constants.FLAT_OUTPUT: True
    }

    pairwise_cnn = ConvEncoder(cnn_config, logger)

    mlp_config = {
        Constants.INPUT_SIZE: pairwise_cnn.output_size,
        Constants.NEURONS: [128, 1],
        Constants.USE_BATCH_NORM: False,
        Constants.USE_LAYER_NORM: False,
        Constants.ACTIVATION_LAST: False
    }

    dist_mlp = FCEncoder(mlp_config, logger)

    return nn.Sequential(pairwise_cnn, dist_mlp)


encoder = make_pairwise_encoder()
target_encoder = make_pairwise_encoder()

sim = Sim(encoder, target_encoder, {
    Constants.GAMMA: 0.9
})

states = torch.rand((10, 3, 50, 50), dtype=torch.float32)
next_states = torch.rand((10, 3, 50, 50), dtype=torch.float32)
actions = torch.zeros((10,), dtype=torch.int32)
rewards = torch.rand((10,), dtype=torch.float32)

print(sim.loss(states, actions, rewards, next_states, 0.1))
