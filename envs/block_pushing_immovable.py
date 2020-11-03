from envs.block_pushing import BlockPushing


class BlockPushingImmovable(BlockPushing):

    def __init__(self, width=5, height=5, render_type='cubes', num_objects=5,
                 seed=None):

        super(BlockPushingImmovable, self).__init__(
            width=width, height=height, render_type=render_type, num_objects=num_objects, seed=seed
        )

    def translate(self, obj_id, offset):
        """"Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """

        # the first two objects are immovable
        if obj_id in [0, 1]:
            return False

        if not self.valid_move(obj_id, offset):
            return False

        self.objects[obj_id][0] += offset[0]
        self.objects[obj_id][1] += offset[1]

        return True
