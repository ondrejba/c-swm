import matplotlib.pyplot as plt
from envs.block_pushing import BlockPushing
import utils
import matplotlib

matplotlib.use("TkAgg")


def main():

    env = BlockPushing(render_type="cubes", background=BlockPushing.BACKGROUND_DETERMINISTIC)
    env.reset()

    while True:

        img = env.render()
        plt.imshow(utils.css_to_ssc(img))
        plt.show()

        while True:
            x = input("action: ")
            try:
                x = int(x)
            except Exception:
                continue
            if x < 9:
                break

        if x == 8:
            env.reset()
            continue

        env.step(x)


main()
