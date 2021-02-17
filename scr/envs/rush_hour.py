import matplotlib.pyplot as plt

#from envs.block_pushing import BlockPushing
from envs.rush_hour import RushHour

import utils
import matplotlib

matplotlib.use("TkAgg")


def main():

    env = RushHour(render_type="shapes")
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
            if x < 20:
                break

        env.step(x)


main()
