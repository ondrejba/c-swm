from gym.envs.registration import register

register(
    'ShapesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesCursorTrain-v0',
    entry_point='envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesCursorEval-v0',
    entry_point='envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    'CubesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)
