import argparse
import torch
import utils
import os
import pickle
import matplotlib.pyplot as plt


from torch.utils import data
import numpy as np
from collections import defaultdict

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
    encoder=args.encoder).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

all_states = []

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0]

        state = model.obj_encoder(model.obj_extractor(obs))
        all_states.append(state.cpu().numpy())

all_states = np.concatenate(all_states, axis=0)

# topk = [1, 5, 10]
topk = [1]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0

pred_states = []
next_states = []

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0]
        next_obs = observations[-1]

        state = model.obj_encoder(model.obj_extractor(obs))
        next_state = model.obj_encoder(model.obj_extractor(next_obs))

        pred_state = state
        for i in range(args_eval.num_steps):
            pred_trans = model.transition_model(pred_state, actions[i])
            pred_state = pred_state + pred_trans

        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())

    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)

    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions
    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    dist_matrix = utils.pairwise_distance_matrix(
        next_state_flat, pred_state_flat)

    #num_digits = 3
    #dist_matrix = (dist_matrix * 10**num_digits).round() / (10**num_digits)
    #dist_matrix = dist_matrix.float()

    dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
    dist_matrix_augmented = torch.cat(
        [dist_matrix_diag, dist_matrix], dim=1)

    # Workaround to get a stable sort in numpy.
    dist_np = dist_matrix_augmented.numpy()
    indices = []
    for row in dist_np:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)
    indices = torch.from_numpy(indices).long()

    print('Processed {} batches of size {}'.format(
        batch_idx + 1, args.batch_size))

    labels = torch.zeros(
        indices.size(0), device=indices.device,
        dtype=torch.int64).unsqueeze(-1)

    num_samples += full_size
    print('Size of current topk evaluation batch: {}'.format(
        full_size))

    for k in topk:
        match = indices[:, :k] == labels
        num_matches = match.sum()
        hits_at[k] += num_matches.item()

    match = indices == labels
    _, ranks = match.max(1)

    indices = indices.cpu().numpy().astype(np.int)
    first_indices = indices[:, 0]

    for i in range(labels.size(0)):

        # real next state
        rns = next_state_cat[i]

        # pred next state
        pns = pred_state_cat[i]

        # matched next state
        if first_indices[i] == 0:
            print("good")
            mns = next_state_cat[i]
        else:
            print("bad")
            index = first_indices[i] - 1
            mns = next_state_cat[index]

        print(indices[i, :10])
        print(dist_matrix_augmented[i, indices[i, :10]])

        for j in range(5):
            plt.subplot(3, 5, 1 + j)
            plt.scatter(all_states[:, j, 0], all_states[:, j, 1])
            plt.scatter(rns[j, 0], rns[j, 1])

            plt.subplot(3, 5, 6 + j)
            plt.scatter(all_states[:, j, 0], all_states[:, j, 1])
            plt.scatter(pns[j, 0], pns[j, 1])

            plt.subplot(3, 5, 11 + j)
            plt.scatter(all_states[:, j, 0], all_states[:, j, 1])
            plt.scatter(mns[j, 0], mns[j, 1])

        plt.show()
