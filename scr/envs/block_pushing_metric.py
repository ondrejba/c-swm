import numpy as np
import matplotlib.pyplot as plt
from envs.block_pushing_metric import BlockPushingMetric
import utils
import matplotlib

matplotlib.use("TkAgg")





def main():

    env = BlockPushingMetric(
        render_type="shapes", background=BlockPushingMetric.BACKGROUND_DETERMINISTIC, num_objects=2, num_colors=5
    )
    env.compute_metric()

    env.metric[list(range(env.num_states)), list(range(env.num_states))] = 1.0

    for idx1 in range(env.num_states):

        dists = env.metric[idx1, :]
        idx2 = np.argmin(dists)

        s1 = env.all_states[idx1]
        s2 = env.all_states[idx2]

        env.load_state_new_(s1)
        env.background_index = s1[-1]
        i1 = env.render()
        env.load_state_new_(s2)
        env.background_index = s2[-1]
        i2 = env.render()

        print(env.metric[idx1, idx2], env.metric[idx1, idx2])

        plt.subplot(1, 2, 1)
        plt.imshow(utils.css_to_ssc(i1))
        plt.subplot(1, 2, 2)
        plt.imshow(utils.css_to_ssc(i2))
        plt.show()


main()
