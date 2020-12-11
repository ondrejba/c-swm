import itertools
import numpy as np
from envs.block_pushing import BlockPushing


class BlockPushingMetric(BlockPushing):

    def __init__(self, width=5, height=5, render_type='cubes', num_objects=3,
                 seed=None, immovable=False, immovable_fixed=False, opposite_direction=False,
                 background=BlockPushing.BACKGROUND_WHITE, num_colors=1, reward_num_goals=10,
                 all_goals=False):

        super(BlockPushingMetric, self).__init__(
            width, height, render_type, num_objects, seed, immovable, immovable_fixed,
            opposite_direction=opposite_direction, background=background, num_colors=num_colors
        )

        # otherwise the memory will explode
        assert num_objects <= 3

        self.num_colors = num_colors
        self.reward_num_goals = reward_num_goals
        self.all_goals = all_goals

        self.all_states = []
        self.next_states = {i: [] for i in range(self.num_actions)}
        self.rewards = {i: [] for i in range(self.num_actions)}
        self.num_states = None
        self.metric = None

        self.enumerate_states_()
        self.set_rewards_and_next_states_()

    def compute_metric(self, tolerance=0.001, max_steps=1000):

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

        goals = np.random.choice(list(range(len(self.all_states))), size=self.reward_num_goals, replace=False)

        for state in self.all_states:

            for action in list(range(self.num_actions)):

                self.load_state_new_(state)
                self.step_no_render(action)
                next_state = self.get_state_new_()
                next_state_index = self.all_states.index(next_state)
                reward = [float(next_state_index == goal) for goal in goals]

                self.rewards[action].append(reward)
                self.next_states[action].append(self.all_states.index(next_state))

        for key in self.rewards.keys():
            self.rewards[key] = np.array(self.rewards[key])
            self.next_states[key] = np.array(self.next_states[key])

    def initialize_metric_(self):

        self.metric = np.zeros((self.num_states, self.num_states), dtype=np.float32)

    def iterate_metric_(self):

        best = None

        for a in range(self.num_actions):

            r_dist = np.zeros((self.num_states, self.num_states), dtype=np.float32)

            if self.all_goals:
                # if every state is a goal than the reward distance is one between non-identical states
                r_dist = np.ones_like(r_dist)
                r_dist[list(range(self.num_states)), list(range(self.num_states))] = 0.0
            else:
                # if there are multiple reward functions we iterate over them to save memory
                for i in range(self.reward_num_goals):

                    r = self.rewards[a][:, i]
                    r1 = r[:, None].repeat(self.num_states, axis=1)
                    r2 = r[None, :].repeat(self.num_states, axis=0)

                    tmp_r_dist = np.abs(r1 - r2)
                    r_dist = np.max([r_dist, tmp_r_dist], axis=0)

                # python dealloc is not doing its job somewhere
                del r1
                del r2
                del tmp_r_dist

            sp = self.next_states[a]
            sp1 = sp[:, None].repeat(self.num_states, axis=1)
            sp2 = sp[None, :].repeat(self.num_states, axis=0)

            sp1 = sp1.reshape(-1)
            sp2 = sp2.reshape(-1)

            d = self.metric[sp1, sp2]
            d = d.reshape((self.num_states, self.num_states))

            del sp1
            del sp2

            new_metric = 0.1 * r_dist + 0.9 * d

            if best is None:
                best = new_metric
            else:
                best = np.max([best, new_metric], axis=0)

        diff = np.max(np.abs(self.metric - best))
        self.metric = best
        return diff

    def load_state_new_(self, state_id):

        # ignore color information
        state_id = state_id[:self.num_objects]
        # parse position
        self.objects = [[c // self.width, c % self.width] for c in state_id]

    def get_state_new_(self):

        return [x * self.width + y for x, y in self.objects]

    def get_state_id(self):

        return self.all_states.index(self.get_state_new_())
