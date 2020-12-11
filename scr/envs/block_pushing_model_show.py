import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from envs.block_pushing import BlockPushing
from train_sim import make_pairwise_encoder
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

    env = BlockPushing(render_type="shapes", background=BlockPushing.BACKGROUND_DETERMINISTIC)

    bisim_model = make_pairwise_encoder()
    bisim_model.load_state_dict(torch.load(args.model_path))
    bisim_model.to(args.device)

    for _ in range(10):

        env.reset()
        base_image = env.render()

        color_images = []
        for i in range(4):
            env.background_index = i + 1
            color_images.append(env.render())

        random_images = []
        for i in range(10):
            env.reset()
            env.background_index = np.random.randint(5)
            random_images.append(env.render())

        all_other_images = [base_image] + color_images + random_images

        if args.plot:
            plot_all_images(all_other_images)

        pairs_first = pair(base_image, all_other_images, base_first=True)
        pairs_second = pair(base_image, all_other_images, base_first=False)

        pairs_first = stack_to_pt(pairs_first, args.device)
        pairs_second = stack_to_pt(pairs_second, args.device)

        dists_first = bisim_model(pairs_first)[:, 0].cpu().numpy()
        dists_second = bisim_model(pairs_second)[:, 0].cpu().numpy()

        print("[base_image, base_image]: {:s}".format(str(dists_first[0])))

        print("[base_image, other_image]")
        print("different colors: {:s}".format(str(dists_first[1:5])))
        print("different states: {:s}".format(str(dists_first[5:])))

        print("[other_image, base_image]")
        print("different colors: {:s}".format(str(dists_second[1:5])))
        print("different states: {:s}".format(str(dists_second[5:])))


parser = argparse.ArgumentParser()

parser.add_argument("model_path")
parser.add_argument("device")
parser.add_argument("--plot", default=False, action="store_true")

parsed = parser.parse_args()
main(parsed)
