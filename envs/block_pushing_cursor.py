import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import skimage
from gym import spaces
from envs.block_pushing import square, triangle, fig2rgb_array, render_cubes, BlockPushing
import utils


class BlockPushingCursor(BlockPushing):

    ACTIONS = {
        0: "CURSOR_UP",
        1: "CURSOR_RIGHT",
        2: "CURSOR_DOWN",
        3: "CURSOR_LEFT",
        4: "OBJ_UP",
        5: "OBJ_RIGHT",
        6: "OBJ_DOWN",
        7: "OBJ_LEFT"
    }

    def __init__(self, width=5, height=5, render_type='cubes', num_objects=5,
                 seed=None, immovable=False):

        super(BlockPushingCursor, self).__init__(width=width, height=height, render_type=render_type,
                                                 num_objects=num_objects, seed=seed, immovable=immovable)

        # overwrite action space
        self.num_actions = 8
        self.action_space = spaces.Discrete(self.num_actions)

        # get color for cursor
        self.colors = utils.get_colors(num_colors=max(9, self.num_objects + 1))

        # initialize cursor outside of the env
        self.cursor = [-1, -1]

    def render(self):

        assert self.render_type in ["shapes"]

        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, pos in enumerate(self.objects):
            if idx % 3 == 0:
                rr, cc = skimage.draw.circle(
                    pos[0] * 10 + 5, pos[1] * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]
            elif idx % 3 == 1:
                rr, cc = triangle(
                    pos[0] * 10, pos[1] * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]
            else:
                rr, cc = square(
                    pos[0] * 10, pos[1] * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]

        # draw cursor
        rr, cc = square(
            self.cursor[0] * 10 + 2.5, self.cursor[1] * 10 + 2.5, 5, im.shape)
        im[rr, cc, :] = self.colors[self.num_objects][:3]

        return im.transpose([2, 0, 1])

    def reset(self):

        self.reset_objects_()
        self.reset_cursor_()

        return (self.get_state(), self.render())

    def step(self, action: int):

        assert action in self.ACTIONS.keys()

        done = False
        reward = 0

        # first four actions move the cursor, next four move an object below the cursor
        directions = action % 4

        if action < 4:
            self.translate_cursor_(self.DIRECTIONS[directions])
        else:
            self.translate_cursor_and_object_below_(self.DIRECTIONS[directions])

        state_obs = (self.get_state(), self.render())

        return state_obs, reward, done, None

    def reset_cursor_(self):
        # randomize cursor position
        self.cursor = [
            np.random.choice(np.arange(self.width)),
            np.random.choice(np.arange(self.height))
        ]

    def translate_cursor_(self, offset):

        old_pos = self.cursor
        new_pos = [p + o for p, o in zip(old_pos, offset)]

        if not self.valid_pos_cursor_(new_pos):
            return

        self.cursor[0] += offset[0]
        self.cursor[1] += offset[1]

    def translate_cursor_and_object_below_(self, offset):

        obj_id_below = None

        for idx, obj in enumerate(self.objects):
            if obj == self.cursor:
                obj_id_below = idx
                break

        if obj_id_below is None:
            return

        if not self.translate(obj_id_below, offset):
            return False

        # object can be moved, move the cursor as well
        self.translate_cursor_(offset)
        return True

    def valid_pos_cursor_(self, pos):

        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        return True
