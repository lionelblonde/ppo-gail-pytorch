from copy import deepcopy

import numpy as np

from gym import spaces
from pycolab.examples.research.box_world import box_world
from pycolab.examples.classics import cliff_walk
from pycolab import ascii_art

from helpers import pycolab_gymify


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def rgbify_dict(color_dict):
    """Rescale scalar from [0, 999] interval to [0, 255] """
    return {k: tuple([int(c / 999 * 255) for c in list(v)])
            for k, v in color_dict.items()}


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Create Pycolab environments.

class BoxWorldEnv(pycolab_gymify.PyColabEnv):
    """Box-world environment.
    The environment was first introduced in https://arxiv.org/pdf/1806.01830.pdf,
    for Relational Reinforcement Learning.
    """

    def __init__(self, default_reward=0.):  # task too hard for -1 as default (reward hacking)
        """The agent 'only actually move[s] if action is one of the 4 directions of movement'
        (comment found at: pycolab/examples/research/box_world/box_world.py#L166). We could
        ass a fifth action to enable the agent to perform a no-op action, but since only the
        agent move in this environment, there is no use for it. Thus, in accordance to
        the pycolab environment, the action space is defined as `range(4)`.
        (Note, any other action would act as a no-op.)

        `max_iterations` is not needed here (but signature is preserved nonetheless) since the
        pycolab environment already set the episode termination horizon with `max_num_steps`.

        For grid_size=12 (as pycolab's default), resize_scale=6 gives a render of size 84x84.
        For grid_size=12 (as pycolab's default), resize_scale=16 gives a render of size 224x224.
        """
        super(BoxWorldEnv, self).__init__(max_iterations=np.infty,
                                          default_reward=default_reward,
                                          action_space=spaces.Discrete(4),
                                          delay=30,
                                          resize_scale=16)

    def make_game(self):
        """Note, those are the settings from the paper."""
        return box_world.make_game(grid_size=12,
                                   solution_length=(1, 2, 3, 4),
                                   num_forward=(0, 1, 2, 3, 4),
                                   num_backward=(0,),
                                   branch_length=1,
                                   random_state=None,
                                   max_num_steps=120)

    def make_colors(self):
        """Return the color dictionary defined in the pycolab environment.
        Note, need to transform it to RGB format for proper rendering.
        """
        color_dict = deepcopy(box_world.OBJECT_COLORS)
        return rgbify_dict(color_dict)


class CliffWalkEnv(pycolab_gymify.PyColabEnv):
    """Classic cliff-walk game."""

    def __init__(self, max_iterations, default_reward=-1.):
        super(CliffWalkEnv, self).__init__(max_iterations=max_iterations,
                                           default_reward=default_reward,
                                           action_space=spaces.Discrete(4),
                                           delay=30,
                                           resize_scale=24)

    def make_game(self):
        """Reimplemention of the game map."""
        # We modify the game art to make the cliff section visual discernible.
        BOOTLEG_GAME_ART = ['......',
                            '......',
                            'Pxxxx.']
        return ascii_art.ascii_art_to_game(BOOTLEG_GAME_ART,
                                           what_lies_beneath='.',
                                           sprites={'P': cliff_walk.PlayerSprite})

    def make_colors(self):
        return {'.': (192, 192, 192),
                'P': (127, 0, 255),
                'x': (0, 0, 0)}


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Create environment maker.

def make_pycolab(env_id):
    if env_id == 'BoxWorld-v0':
        return BoxWorldEnv()
    elif env_id == 'CliffWalk-v0':
        return CliffWalkEnv(max_iterations=150)
    else:
        pass
