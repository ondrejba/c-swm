# obs, action, next_obs = data_batch
# obs, next_obs: [1024,3,50,50]
# objs: [1024,5,5,5]

##To do: make x and rot_x inputs to constrastive loss.
from argparse import Namespace
from torch.utils import data
import utils
import modules
import torch
import numpy as np
import matplotlib.pyplot as mpl

args = Namespace()
args.dataset = 'data/shapes_train.h5'
args.batch_size = 5
args.embedding_dim = 2
args.hidden_dim = 256
args.action_dim = 4
args.num_objects = 3
args.sigma = 0.5
args.hinge = 1.
args.ignore_action = False
args.copy_action = False
args.split_mlp = False
args.same_ep_neg = False
args.only_same_ep_neg = False
args.immovable_bit = False
args.split_gnn = False
args.encoder = 'small'
args.rot = True

# device = torch.device('cuda' if args.cuda else 'cpu')
device = 'cpu'  #Debug

model = modules.ContrastiveSWM(embedding_dim=args.embedding_dim,
                               hidden_dim=args.hidden_dim,
                               action_dim=args.action_dim,
                               input_dims=(args.batch_size, 3, 50, 50),
                               num_objects=args.num_objects,
                               sigma=args.sigma,
                               hinge=args.hinge,
                               ignore_action=args.ignore_action,
                               copy_action=args.copy_action,
                               split_mlp=args.split_mlp,
                               same_ep_neg=args.same_ep_neg,
                               only_same_ep_neg=args.only_same_ep_neg,
                               immovable_bit=args.immovable_bit,
                               split_gnn=args.split_gnn,
                               encoder=args.encoder,
                               rot=args.rot).to(device)


def contrastive_loss(objs, action, next_objs):

    state = model.obj_encoder(objs)
    next_state = model.obj_encoder(next_objs)

    # Sample negative state across episodes at random
    batch_size = state.size(0)
    perm = np.random.permutation(batch_size)
    neg_state = state[perm]

    model.pos_loss = model.energy(state, action, next_state)
    zeros = torch.zeros_like(model.pos_loss)

    model.pos_loss = model.pos_loss.mean()
    model.neg_loss = torch.max(
        zeros, model.hinge -
        model.energy(state, action, neg_state, no_trans=True)).mean()

    # if model.same_ep_neg:
    #     ep_size = state.size(1)
    #     neg_state = state.clone()

    #     # same perm for all batches
    #     #perm = np.random.permutation(np.arange(ep_size))
    #     #neg_state[:, :] = neg_state[:, perm]

    #     # different perm for each batch
    #     perm = np.stack([
    #         np.random.permutation(np.arange(ep_size)) for _ in range(batch_size)
    #     ],
    #                     axis=0)
    #     indices = np.arange(batch_size)[:, np.newaxis].repeat(ep_size, axis=1)
    #     neg_state[:, :] = neg_state[indices, perm]

    #     if model.only_same_ep_neg:
    #         # overwrite the negative loss calculated above
    #         model.neg_loss = torch.max(
    #             zeros, model.hinge -
    #             model.energy(state, action, neg_state, no_trans=True)).mean()
    #     else:
    #         # average negative loss over in-episode and out-of-episode samples
    #         model.neg_loss += torch.max(
    #             zeros, model.hinge -
    #             model.energy(state, action, neg_state, no_trans=True)).mean()
    #         model.neg_loss /= 2

    loss = model.pos_loss + model.neg_loss

    return model.pos_loss, model.neg_loss


# ["up", "right", "down", "left"]  CCW => -1
objs = torch.rand(args.batch_size, args.num_objects, 5, 5)
actions = torch.tensor([2] * args.batch_size)
next_objs = torch.rand(args.batch_size, args.num_objects, 5, 5)

# mpl.matshow(next_objs[0, 0, ...])
# Non rotated loss
pos_loss, neg_loss = contrastive_loss(objs, actions, next_objs)

# Rotated objs, next_objs
rot_objs = model.obj_encoder.rot90(objs)
rot_actions = torch.tensor([1] * args.batch_size)
rot_next_objs = model.obj_encoder.rot90(next_objs)

# Visualize objs and rot_objs
mpl.matshow(objs[0, 0, ...])
mpl.matshow(rot_objs[0, 0, ...])
mpl.matshow(objs[0, 1, ...])
mpl.matshow(rot_objs[0, 1, ...])

rot_pos_loss, rot_neg_loss = contrastive_loss(rot_objs, rot_actions,
                                              rot_next_objs)
print(f"Non-rot: pos={pos_loss}, neg={neg_loss}")
print(f"Rot: pos={rot_pos_loss}, neg={rot_neg_loss}")

# negative loss is different because there is one negative sample for each state
# different permutations used in neg_sampling gives different loss values
