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
    'ShapesImmovableTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesImmovableEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesOppositeTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'opposite_direction': True},
)

register(
    'ShapesOppositeEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'opposite_direction': True},
)

register(
    'ShapesImmovableFixedTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'immovable': True, 'immovable_fixed': True},
)

register(
    'ShapesImmovableFixedEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'immovable': True, 'immovable_fixed': True},
)

register(
    'ShapesCursorImmovableTrain-v0',
    entry_point='envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesCursorImmovableEval-v0',
    entry_point='envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'immovable': True},
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

register(
    'CubesCursorTrain-v0',
    entry_point='envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesCursorEval-v0',
    entry_point='envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesImmovableTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes', 'immovable': True},
)

register(
    'CubesImmovableEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes', 'immovable': True},
)
