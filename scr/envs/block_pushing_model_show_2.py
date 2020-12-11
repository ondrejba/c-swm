import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from train_sim import make_pairwise_encoder
import utils
matplotlib.use("TkAgg")


def pair(base_image, other_images, base_first=True):

    ret = []

    for image in other_images:

        if base_first:
            tmp_image = np.concatenate([base_image, image], axis=0)
        else:
            tmp_image = np.concatenate([image, base_image], axis=0)

        ret.append(tmp_image)

    return ret


def stack_to_pt(images, device):

    images = np.stack(images, axis=0)
    return torch.tensor(images, dtype=torch.float32, device=device)


def plot_all_images(images):

    for idx, image in enumerate(images):

        plt.subplot(1, len(images), 1 + idx)
        plt.imshow(image.transpose((1, 2, 0)))

    plt.show()


@torch.no_grad()
def main(args):

    dataset = "data/shapes_bg_deterministic_train.h5"
    batch_size = 1024

    bisim_model = make_pairwise_encoder()
    bisim_model.load_state_dict(torch.load(args.model_path))
    bisim_model.to(args.device)

    dataset = utils.StateTransitionsDatasetStateIds(hdf5_file=dataset)

    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, data_batch in enumerate(train_loader):

        data_batch = [tensor.to(args.device) for tensor in data_batch]
        obs, action, next_obs, state_ids, next_state_ids = data_batch

        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        neg_obs = obs[perm]

        stack = torch.cat([obs, neg_obs], dim=1)
        dists = bisim_model(stack)[:, 0].detach()
        print(dists)

        if batch_idx > 10:
            break


parser = argparse.ArgumentParser()

parser.add_argument("model_path")
parser.add_argument("device")
parser.add_argument("--plot", default=False, action="store_true")

parsed = parser.parse_args()
main(parsed)
