import argparse
import torch
import utils
import datetime
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import logging
from train_sim import make_pairwise_encoder

from torch.utils import data
import torch.nn.functional as F

import modules


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extractor CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')
parser.add_argument('--split-mlp', action='store_true', default=False,
                    help='Create two MLPs, one for movable and the other for immovable objects.')
parser.add_argument('--split-gnn', action='store_true', default=False)
parser.add_argument('--immovable-bit', action='store_true', default=False)
parser.add_argument('--same-ep-neg', action='store_true', default=False)
parser.add_argument('--only-same-ep-neg', action='store_true', default=False)
parser.add_argument('--no-loss-first-two', action='store_true', default=False)
parser.add_argument('--bisim', action='store_true', default=False)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--custom-neg', default=False, action='store_true')
parser.add_argument('--bisim-metric-path')
parser.add_argument('--bisim-eps', type=float)
parser.add_argument('--bisim-model-path')
parser.add_argument('--next-state-neg', default=False, action="store_true")
parser.add_argument('--nl-type', default=0, type=int)
parser.add_argument('--coord-grid', default=False, action='store_true')

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')
loss_file = os.path.join(save_folder, 'loss.pdf')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

if args.bisim:
    if args.custom_neg:
        dataset = utils.StateTransitionsDatasetStateIdsNegs(
            hdf5_file=args.dataset)
    else:
        dataset = utils.StateTransitionsDatasetStateIds(
            hdf5_file=args.dataset)
else:
    assert not args.custom_neg
    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset)

train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

# maybe load bisim metric and turn it into torch tensor on the selected device
bisim_metric = None
if args.bisim_metric_path is not None:
    bisim_metric = torch.tensor(np.load(args.bisim_metric_path), dtype=torch.float32, device=device)

# maybe load bisim model
bisim_model = None
if args.bisim_model_path is not None:
    bisim_model = make_pairwise_encoder()
    bisim_model.load_state_dict(torch.load(args.bisim_model_path))

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
    gamma=args.gamma,
    bisim_metric=bisim_metric,
    bisim_eps=args.bisim_eps,
    next_state_neg=args.next_state_neg,
    nl_type=args.nl_type,
    encoder=args.encoder,
    use_coord_grid=args.coord_grid).to(device)

model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

# avoid re-initializing the model and adding to the trainable parameters list
if bisim_model is not None:
    model.bisim_model = bisim_model
    model.bisim_model.to(device)

if args.decoder:
    if args.encoder == 'large':
        decoder = modules.DecoderCNNLarge(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'medium':
        decoder = modules.DecoderCNNMedium(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'small':
        decoder = modules.DecoderCNNSmall(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    decoder.apply(utils.weights_init)
    optimizer_dec = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)


# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9
losses = []

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()

        if args.decoder:
            assert not args.bisim # not implemented
            optimizer_dec.zero_grad()
            obs, action, next_obs = data_batch
            objs = model.obj_extractor(obs)
            state = model.obj_encoder(objs)

            rec = torch.sigmoid(decoder(state))
            loss = F.binary_cross_entropy(
                rec, obs, reduction='sum') / obs.size(0)

            next_state_pred = state + model.transition_model(state, action)
            next_rec = torch.sigmoid(decoder(next_state_pred))
            next_loss = F.binary_cross_entropy(
                next_rec, next_obs,
                reduction='sum') / obs.size(0)
            loss += next_loss
        else:
            loss = model.contrastive_loss(*data_batch)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        losses.append(loss.item())

        if args.decoder:
            optimizer_dec.step()

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_file)

plt.subplot(2, 1, 1)
plt.plot(losses)
plt.subplot(2, 1, 2)
plt.plot(losses)
plt.yscale("log")
plt.savefig(loss_file)
