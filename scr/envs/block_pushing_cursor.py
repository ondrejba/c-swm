import matplotlib.pyplot as plt
from envs.block_pushing import BlockPushing
from envs.block_pushing_cursor import BlockPushingCursor
import utils
import matplotlib

#matplotlib.use("TkAgg")


def main():

    env = BlockPushingCursor(render_type="shapes")
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
            if x < 8:
                break

        env.step(x)


main()
