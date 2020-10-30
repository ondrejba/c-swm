import h5py
import matplotlib.pyplot as plt
import argparse


PATH = "data/cubes_train_tiny.h5"
DIRECTIONS = ["up", "right", "down", "left"]


def css_to_ssc(image):
    return image.transpose((1, 2, 0))


def action_to_text(action):
    direction = action % 4
    obj = action // 4
    return "move object {:d} {:s}".format(obj, DIRECTIONS[direction])


def main(args):

    data = h5py.File(args.path, "r")
    episodes = [int(key) for key in data.keys()]
    episodes = list(sorted(episodes))

    for episode in episodes:

        ep_data = data[str(episode)]
        num_steps = ep_data["obs"].shape[0]

        for step in range(num_steps):

            obs = ep_data["obs"][step]
            action = ep_data["action"][step]
            next_obs = ep_data["next_obs"][step]

            print(action_to_text(action))

            if args.balls:
                plt.subplot(2, 2, 1)
                plt.imshow(css_to_ssc(obs)[:,:,0:3])
                plt.axis("off")
                plt.subplot(2, 2, 2)
                plt.imshow(css_to_ssc(obs)[:,:,3:6])
                plt.axis("off")
            else:
                plt.subplot(1, 2, 1)
                plt.imshow(css_to_ssc(obs))
                plt.axis("off")
            
            
            if args.balls:
                plt.subplot(2, 2, 3)
                plt.imshow(css_to_ssc(next_obs)[:,:,0:3])
                plt.axis("off")
                plt.subplot(2, 2, 4)
                plt.imshow(css_to_ssc(next_obs)[:,:,3:6])
                plt.axis("off")
            else:
                plt.subplot(1, 2, 2)
                plt.imshow(css_to_ssc(next_obs))
                plt.axis("off")
            plt.show()


p = argparse.ArgumentParser()
p.add_argument("--path")
p.add_argument("--balls",action="store_true")
args = p.parse_args()

main(args)
