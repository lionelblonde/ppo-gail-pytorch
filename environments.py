# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MuJoCo environments.

MUJOCO_ROBOTS = [
    'InvertedPendulum',
    'InvertedDoublePendulum',
    'Reacher',
    'Hopper',
    'HalfCheetah',
    'Walker2d',
    'Ant',
    'Humanoid',
]

MUJOCO_ENVS = ["{}-v2".format(name) for name in MUJOCO_ROBOTS]
MUJOCO_ENVS.extend(["{}-v3".format(name) for name in MUJOCO_ROBOTS])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DM Control environments.

DMC_ROBOTS = [
    'Hopper-Hop',
    'Cheetah-Run',
    'Walker-Walk',
    'Walker-Run',

    'Stacker-Stack_2',
    'Stacker-Stack_4',

    'Humanoid-Walk',
    'Humanoid-Run',
    'Humanoid-Run_Pure_State',

    'Humanoid_CMU-Stand',
    'Humanoid_CMU-Run',

    'Quadruped-Walk',
    'Quadruped-Run',
    'Quadruped-Escape',
    'Quadruped-Fetch',

    'Dog-Run',
    'Dog-Fetch',
]

DMC_ENVS = ["{}-Feat-v0".format(name) for name in DMC_ROBOTS]
DMC_ENVS.extend(["{}-Pix-v0".format(name) for name in DMC_ROBOTS])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Safety Gym environments.

SAFETY_ENVS = [
    'Safexp-PointGoal0',
    'Safexp-PointGoal1',
    'Safexp-PointGoal2',
    'Safexp-CarGoal0',
    'Safexp-CarGoal1',
    'Safexp-CarGoal2',
    'Safexp-DoggoGoal0',
    'Safexp-DoggoGoal1',
    'Safexp-DoggoGoal2',

    'Safexp-PointButton0',
    'Safexp-PointButton1',
    'Safexp-PointButton2',
    'Safexp-CarButton0',
    'Safexp-CarButton1',
    'Safexp-CarButton2',
    'Safexp-DoggoButton0',
    'Safexp-DoggoButton1',
    'Safexp-DoggoButton2',

    'Safexp-PointPush0',
    'Safexp-PointPush1',
    'Safexp-PointPush2',
    'Safexp-CarPush0',
    'Safexp-CarPush1',
    'Safexp-CarPush2',
    'Safexp-DoggoPush0',
    'Safexp-DoggoPush1',
    'Safexp-DoggoPush2',
]

SAFETY_ENVS = ["{}-v0".format(name) for name in SAFETY_ENVS]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Atari games.

ATARI_GAMES = list(map(lambda name: ''.join([g.capitalize() for g in name.split('_')]), [
    'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout',
    'carnival', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon',
]))

ATARI_ENVS = ["{}NoFrameskip-v4".format(name) for name in ATARI_GAMES]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PyColab environments.

PYCOLAB_ENVS = [
    'BoxWorld',
    'CliffWalk',
]

PYCOLAB_ENVS = ["{}-v0".format(name) for name in PYCOLAB_ENVS]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Aggregate the environments

BENCHMARKS = {
    'mujoco': MUJOCO_ENVS,
    'dmc': DMC_ENVS,
    'safety': SAFETY_ENVS,
    'atari': ATARI_ENVS,
    'pycolab': PYCOLAB_ENVS,
}
