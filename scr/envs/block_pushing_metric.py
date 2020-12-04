import numpy as np
import matplotlib.pyplot as plt
from envs.block_pushing_metric import BlockPushingMetric
import utils
import matplotlib

matplotlib.use("TkAgg")





def main():

    env = BlockPushingMetric(render_type="shapes", background=BlockPushingMetric.BACKGROUND_DETERMINISTIC, num_objects=2)
    env.compute_metric()


main()
