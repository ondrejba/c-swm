import argparse
import torch
import utils
import os
import pickle
import matplotlib.pyplot as plt


from torch.utils import data
import numpy as np
from collections import defaultdict
from train_sim import make_pairwise_encoder

import modules

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

args_eval = parser.parse_args()


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 100
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.PathDataset(
    hdf5_file=args.dataset, path_length=args_eval.num_steps)
eval_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0][0].size()

model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
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
    no_loss_first_two=args.no_loss_first_two,
    bisim_model=make_pairwise_encoder() if args.bisim_model_path else None,
    encoder=args.encoder).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

# topk = [1, 5, 10]
topk = [1]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0

pred_states = []
next_states = []

states = []

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0]

        state = model.obj_encoder(model.obj_extractor(obs))
        states.append(state.cpu().numpy())

all_states = np.concatenate(states, axis=0)

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0]
        next_obs = observations[-1]

        state_ext = model.obj_extractor(obs)
        state = model.obj_encoder(state_ext)
        state_np = state.cpu().numpy()
        print("object embeddings:", state[0])
        print("action", actions[0][0] // 4, actions[0][0] % 4)

        pred_trans = model.transition_model(state, actions[0])
        pred_state = state + pred_trans
        pred_state_np = pred_state.cpu().numpy()

        num_objects = state_ext.shape[1]

        # current obs | obj 1 | obj 2 | ...
        # next obs | obj 1 | obj 2 | ...
        plt.figure(figsize=(12, 7))

        plt.subplot(3, 7, 1)
        plt.imshow(utils.css_to_ssc(utils.to_np(obs[0])))
        plt.axis("off")

        for i in range(num_objects):

            plt.subplot(3, 7, 2 + i)
            plt.imshow(utils.to_np(state_ext[0, i]))
            plt.axis("off")

        for i in range(num_objects):

            plt.subplot(3, 7, 9 + i)
            plt.scatter(all_states[:, i, 0], all_states[:, i, 1])
            plt.scatter(state_np[0, i, 0], state_np[0, i, 1])

        plt.subplot(3, 7, 15)
        plt.imshow(utils.css_to_ssc(utils.to_np(next_obs[0])))
        plt.axis("off")

        for i in range(num_objects):

            plt.subplot(3, 7, 16 + i)
            plt.scatter(all_states[:, i, 0], all_states[:, i, 1])
            plt.scatter(pred_state_np[0, i, 0], pred_state_np[0, i, 1])

        plt.tight_layout()
        plt.show()
