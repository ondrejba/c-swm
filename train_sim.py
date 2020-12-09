import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils import data
from modules2.ConvEncoder import ConvEncoder
from modules2.FCEncoder import FCEncoder
from models.Sim import Sim
from logger import Logger
from constants import Constants
import utils


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


def train(model, opt, device, train_loader, epochs, log_interval, model_file, beta_exp, update_freq, metric):

    # Train model.
    print('Starting model training...')
    step = 0
    one_minus_beta = 1
    best_abs_error = 1e9
    losses = []
    abs_errors = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_abs_error = 0

        for batch_idx, data_batch in enumerate(train_loader):

            if step > 0 and step % update_freq == 0:

                one_minus_beta *= beta_exp
                model.sync()

            data_batch = [tensor.to(device) for tensor in data_batch]
            opt.zero_grad()

            beta = 1 - one_minus_beta
            ret = model.loss(*data_batch[:3], beta, state_ids=data_batch[3], metric=metric)

            if metric is not None:
                loss, abs_error = ret
            else:
                loss = ret
                abs_error = ret

            loss.backward()
            train_loss += loss.item()
            train_abs_error += abs_error.item()
            opt.step()

            losses.append(loss.item())
            abs_errors.append(abs_error.item())

            if batch_idx % log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item() / len(data_batch[0])))

            step += 1

        avg_loss = train_loss / len(train_loader.dataset)
        avg_error = train_abs_error / len(train_loader.dataset)
        print('====> Epoch: {} Avg loss: {:.6f}, Avg error: {:.6f}'.format(
            epoch, avg_loss, avg_error))

        if avg_error < best_abs_error:
            best_abs_error = avg_error
            torch.save(model.encoder.state_dict(), model_file)

    return losses


def get_paths(save_folder, exp_name):

    save_folder = '{}/{}/'.format(save_folder, exp_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'model.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    loss_file = os.path.join(save_folder, 'loss.pdf')

    return meta_file, model_file, log_file, loss_file


def plot_loss(losses, loss_file):

    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.yscale("log")
    plt.savefig(loss_file)


def main(args):

    meta_file, model_file, log_file, loss_file = get_paths(args.save_folder, args.name)

    device = torch.device('cuda' if args.cuda else 'cpu')

    encoder = make_pairwise_encoder()
    target_encoder = make_pairwise_encoder()

    sim = Sim(encoder, target_encoder, {
        Constants.GAMMA: 0.9
    }).to(device)

    opt = optim.Adam(params=sim.encoder.parameters(), lr=1e-2)

    dataset = utils.StateTransitionsDatasetStateIds(hdf5_file=args.dataset)

    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    bisim_metric = None
    if args.bisim_metric_path is not None:
        bisim_metric = torch.tensor(np.load(args.bisim_metric_path), dtype=torch.float32, device=device)

    epochs = 100
    log_interval = 20
    beta_exp = 0.9
    update_freq = 500

    losses = train(
        sim, opt, device, train_loader, epochs, log_interval, model_file, beta_exp, update_freq, bisim_metric
    )
    plot_loss(losses, loss_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='data/shapes_train.h5',
                        help='Path to replay buffer.')
    parser.add_argument('--name', type=str, default='none',
                        help='Experiment name.')
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size.')
    parser.add_argument('--bisim-metric-path')
    parser.add_argument('--no-cuda', default=False, action='store_true')

    parsed = parser.parse_args()
    parsed.cuda = not parsed.no_cuda
    main(parsed)
