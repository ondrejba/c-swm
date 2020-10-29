import h5py
import matplotlib.pyplot as plt

PATH = "data/shapes_train.h5"
DIRECTIONS = ["up", "right", "down", "left"]


def css_to_ssc(image):
    return image.transpose((1, 2, 0))


def action_to_text(action):
    direction = action % 4
    obj = action // 4
    return "move object {:d} {:s}".format(obj, DIRECTIONS[direction])


def main():

    data = h5py.File(PATH, "r")
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
            plt.subplot(1, 2, 1)
            plt.imshow(css_to_ssc(obs))
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(css_to_ssc(next_obs))
            plt.axis("off")
            plt.show()


main()
