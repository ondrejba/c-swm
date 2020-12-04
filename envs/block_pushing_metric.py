import itertools
import numpy as np
from envs.block_pushing import BlockPushing


class BlockPushingMetric(BlockPushing):

    def __init__(self, width=5, height=5, render_type='cubes', num_objects=3,
                 seed=None, immovable=False, immovable_fixed=False, opposite_direction=False,
                 background=BlockPushing.BACKGROUND_WHITE):

        super(BlockPushingMetric, self).__init__(
            width, height, render_type, num_objects, seed, immovable, immovable_fixed, opposite_direction,
            background
        )

        assert num_objects <= 3

        self.all_states = []
        self.next_states = {i: [] for i in range(self.num_actions)}
        self.rewards = {i: [] for i in range(self.num_actions)}
        self.num_states = None
        self.metric = None

    def compute_metric(self, tolerance=0.001, max_steps=1000):

        self.enumerate_states_()
        self.set_rewards_and_next_states_()
        self.initialize_metric_()

        for i in range(max_steps):

            delta = self.iterate_metric_()
            print(delta)
            if delta < tolerance:
                break

    def save_metric(self, save_path):

        np.save(save_path, self.metric)

    def load_metric(self, load_path):

        self.metric = np.load(load_path)

    def enumerate_states_(self):

        num_pos = self.height * self.width
        states = itertools.permutations(list(range(num_pos)), self.num_objects)
        states = [list(state) for state in states]
        self.all_states = states
        self.num_states = len(self.all_states)

    def set_rewards_and_next_states_(self):

        for state in self.all_states:

            for action in list(range(self.num_actions)):

                self.load_state_new_(state)
                self.step_no_render(action)
                next_state = self.get_state_new_()
                reward = float(state != next_state)
                self.next_states[action].append(self.all_states.index(next_state))
                self.rewards[action].append(reward)

    def initialize_metric_(self):

        self.metric = np.zeros((self.num_states, self.num_states), dtype=np.float32)

    def iterate_metric_(self):

        for key in self.rewards.keys():
            self.rewards[key] = np.array(self.rewards[key])
            self.next_states[key] = np.array(self.next_states[key])

        tmp = []

        for a in range(self.num_actions):

            r = self.rewards[a]
            r1 = r[:, None].repeat(self.num_states, axis=1)
            r2 = r[None, :].repeat(self.num_states, axis=0)

            sp = self.next_states[a]
            sp1 = sp[:, None].repeat(self.num_states, axis=1)
            sp2 = sp[None, :].repeat(self.num_states, axis=0)

            sp1 = sp1.reshape(-1)
            sp2 = sp2.reshape(-1)

            d = self.metric[sp1, sp2]
            d = d.reshape((self.num_states, self.num_states))

            new_metric = 0.1 * np.abs(r1 - r2) + 0.9 * d
            tmp.append(new_metric)

        new_metric = np.max(tmp, axis=0)

        diff = np.max(np.abs(self.metric - new_metric))
        self.metric = new_metric
        return diff

    def load_state_new_(self, state_id):

        self.objects = [[c // self.width, c % self.width] for c in state_id]

    def get_state_new_(self):

        return [x * self.width + y for x, y in self.objects]
