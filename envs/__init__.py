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
    'ShapesNoTrianglesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'no_triangles': True},
)

register(
    'ShapesNoTrianglesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'no_triangles': True},
)

register(
    'ShapesDiamondsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'diamonds': True},
)

register(
    'ShapesDiamondsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'diamonds': True},
)

register(
    'ShapesDiamondsTrain1S-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=1,
    kwargs={'render_type': 'shapes', 'diamonds': True},
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
    'RushHourTrain-v0',
    entry_point='envs.rush_hour:RushHour',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'RushHourEval-v0',
    entry_point='envs.rush_hour:RushHour',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)
