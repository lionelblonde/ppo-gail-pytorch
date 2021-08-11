import argparse
from copy import deepcopy
import os
import sys
import numpy as np
import subprocess
import yaml
from datetime import datetime

from helpers import logger
from helpers.misc_util import zipsame, boolean_flag
from helpers.experiment import uuid as create_uuid


ENV_BUNDLES = {
    'mujoco': {
        'debug': ['Hopper-v3'],
        'eevee': ['InvertedPendulum-v2',
                  'InvertedDoublePendulum-v2'],
        'glaceon': ['Hopper-v3',
                    'Walker2d-v3',
                    'HalfCheetah-v3',
                    'Ant-v3'],
        'humanoid': ['Humanoid-v3'],
        'ant': ['Ant-v3'],
        'suite': ['InvertedDoublePendulum-v2',
                  'Hopper-v3',
                  'Walker2d-v3',
                  'HalfCheetah-v3',
                  'Ant-v3'],
    },
    'dmc': {
        'debug': ['Hopper-Hop-Feat-v0'],
        'flareon': ['Hopper-Hop-Feat-v0',
                    'Walker-Run-Feat-v0'],
        'glaceon': ['Hopper-Hop-Feat-v0',
                    'Cheetah-Run-Feat-v0',
                    'Walker-Run-Feat-v0'],
        'stacker': ['Stacker-Stack_2-Feat-v0',
                    'Stacker-Stack_4-Feat-v0'],
        'humanoid': ['Humanoid-Walk-Feat-v0',
                     'Humanoid-Run-Feat-v0'],
        'cmu': ['Humanoid_CMU-Stand-Feat-v0',
                'Humanoid_CMU-Run-Feat-v0'],
        'quad': ['Quadruped-Walk-Feat-v0',
                 'Quadruped-Run-Feat-v0',
                 'Quadruped-Escape-Feat-v0',
                 'Quadruped-Fetch-Feat-v0'],
        'dog': ['Dog-Run-Feat-v0',
                'Dog-Fetch-Feat-v0'],
    },
    'safety': {
        'debug_nohazards': ['Safexp-PointGoal0-v0'],
        'debug_hazards': ['Safexp-PointGoal1-v0'],
        'point_nohazards': ['Safexp-PointGoal0-v0',
                            'Safexp-PointPush0-v0'],
        'point_hazards': ['Safexp-PointGoal1-v0',
                          'Safexp-PointGoal2-v0',
                          'Safexp-PointPush1-v0',
                          'Safexp-PointPush2-v0'],
        'car_nohazards': ['Safexp-CarGoal0-v0',
                          'Safexp-CarPush0-v0'],
        'car_hazards': ['Safexp-CarGoal1-v0',
                        'Safexp-CarGoal2-v0',
                        'Safexp-CarPush1-v0',
                        'Safexp-CarPush2-v0'],
        'doggo_nohazards': ['Safexp-DoggoGoal0-v0',
                            'Safexp-DoggoPush0-v0'],
        'doggo_hazards': ['Safexp-DoggoGoal1-v0',
                          'Safexp-DoggoGoal2-v0',
                          'Safexp-DoggoPush1-v0',
                          'Safexp-DoggoPush2-v0'],
    },
    'atari': {
        'easy': ['Pong'],
        'normal': ['Qbert',
                   'MsPacman',
                   'SpaceInvaders',
                   'Frostbite',
                   'Freeway',
                   'BeamRider',
                   'Asteroids'],
        'hard_exploration': ['MontezumaRevenge',
                             'Pitfall',
                             'PrivateEye'],
    },
    'pycolab': {
        'boxworld': ['BoxWorld-v0'],
        'cliffwalk': ['CliffWalk-v0'],
    },
}

MEMORY = 32


class Spawner(object):

    def __init__(self, args):
        self.args = args

        # Retrieve config from filesystem
        self.config = yaml.safe_load(open(self.args.config))

        # Check if we need expert demos
        self.need_demos = self.config['meta']['algo'] == 'gail'
        assert not self.need_demos or self.config['offline']
        if self.need_demos:
            self.num_demos = [int(i) for i in self.args.num_demos]
        else:
            self.num_demos = [0]  # arbitrary, only used for dim checking

        # Assemble wandb project name
        self.wandb_project = '-'.join([self.config['logging']['wandb_project'].upper(),
                                       self.args.deployment.upper(),
                                       datetime.now().strftime('%B')[0:3].upper() + f"{datetime.now().year}"])

        # Define spawn type
        self.type = 'sweep' if self.args.sweep else 'fixed'

        # Define the needed memory in GB
        self.memory = MEMORY

        # Write out the boolean arguments (using the 'boolean_flag' function)
        self.bool_args = ['cuda', 'render', 'record', 'layer_norm', 'shared_value',
                          'state_only', 'minimax_only', 'spectral_norm', 'grad_pen', 'one_sided_pen',
                          'wrap_absorb', 'd_batch_norm',
                          'red_batch_norm', 'rnd_explo', 'rnd_batch_norm',
                          'dyn_batch_norm', 'use_purl']

        if self.args.deployment == 'slurm':
            # Translate intuitive 'caliber' into actual duration and partition on the Baobab cluster
            calibers = dict(short='0-06:00:00',
                            long='0-12:00:00',
                            verylong='1-00:00:00',
                            veryverylong='2-00:00:00',
                            veryveryverylong='4-00:00:00')
            self.duration = calibers[self.args.caliber]  # intended KeyError trigger if invalid caliber
            if 'verylong' in self.args.caliber:
                if self.config['resources']['cuda']:
                    self.partition = 'public-gpu'
                else:
                    self.partition = 'public-cpu'
            else:
                if self.config['resources']['cuda']:
                    self.partition = 'shared-gpu'
                else:
                    self.partition = 'shared-cpu'

        # Define the set of considered environments from the considered suite
        self.envs = ENV_BUNDLES[self.config['meta']['benchmark']][self.args.env_bundle]

        if self.need_demos:
            # Create the list of demonstrations associated with the environments
            demo_dir = os.environ['DEMO_DIR']
            self.demos = {k: os.path.join(demo_dir, k) for k in self.envs}

    def copy_and_add_seed(self, hpmap, seed):
        hpmap_ = deepcopy(hpmap)

        # Add the seed and edit the job uuid to only differ by the seed
        hpmap_.update({'seed': seed})

        # Enrich the uuid with extra information
        try:
            out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            gitsha = "gitSHA_{}".format(out.strip().decode('ascii'))
        except OSError:
            pass

        uuid = f"{hpmap['uuid']}.{gitsha}.{hpmap['env_id']}.{hpmap['algo']}_{self.args.num_workers}"
        if self.need_demos:
            uuid += f".demos{str(hpmap['num_demos']).zfill(3)}"
        uuid += f".seed{str(seed).zfill(2)}"

        hpmap_.update({'uuid': uuid})

        return hpmap_

    def copy_and_add_env(self, hpmap, env):
        hpmap_ = deepcopy(hpmap)
        # Add the env and demos
        hpmap_.update({'env_id': env})
        if self.need_demos:
            hpmap_.update({'expert_path': self.demos[env]})
        return hpmap_

    def copy_and_add_num_demos(self, hpmap, num_demos):
        assert self.need_demos
        hpmap_ = deepcopy(hpmap)
        # Add the num of demos
        hpmap_.update({'num_demos': num_demos})
        return hpmap_

    def get_hps(self):
        """Return a list of maps of hyperparameters"""

        # Create a uuid to identify the current job
        uuid = create_uuid()

        # Assemble the hyperparameter map
        if self.args.sweep:
            # Random search
            hpmap = {
                'wandb_project': self.wandb_project,
                'uuid': uuid,
                'cuda': self.config['resources']['cuda'],
                'render': False,
                'record': self.config['logging'].get('record', False),
                'task': self.config['meta']['task'],
                'algo': self.config['meta']['algo'],

                # Training
                'num_timesteps': int(float(self.config.get('num_timesteps', 2e7))),
                'eval_steps_per_iter': self.config.get('eval_steps_per_iter', 10),
                'eval_frequency': self.config.get('eval_frequency', 10),

                # Model
                'perception_stack': self.config['perception_stack'],
                'layer_norm': self.config['layer_norm'],
                'shared_value': self.config['shared_value'],

                # Optimization
                'p_lr': float(np.random.choice([1e-3, 3e-4])),
                'v_lr': float(np.random.choice([3e-3, 1e-3])),
                'lr_schedule': self.config['lr_schedule'],
                'clip_norm': np.random.choice([.5, 1., 20., 40.]),

                # Algorithm
                'rollout_len': np.random.choice([1024, 2048]),
                'optim_epochs_per_iter': np.random.choice([1, 2, 6, 10]),
                'batch_size': np.random.choice([32, 64, 128]),
                'gamma': np.random.choice([0.99, 0.995]),
                'gae_lambda': np.random.choice([0.95, 0.98, 0.99]),
                'eps': np.random.choice([0.1, 0.2, 0.4]),
                'baseline_scale': float(np.random.choice([0.1, 0.3, 0.5])),
                'p_ent_reg_scale': self.config.get('p_ent_reg_scale', 0.),

                # Adversarial imitation
                'g_steps': self.config.get('g_steps', 3),
                'd_steps': self.config.get('d_steps', 1),
                'd_lr': float(self.config.get('d_lr', 3e-4)),
                'state_only': self.config.get('state_only', False),
                'minimax_only': self.config.get('minimax_only', True),
                'd_ent_reg_scale': self.config.get('d_ent_reg_scale', 0.001),
                'spectral_norm': self.config.get('spectral_norm', True),
                'grad_pen': self.config.get('grad_pen', True),
                'grad_pen_type': self.config.get('grad_pen_type', 'wgan'),
                'one_sided_pen': self.config.get('one_sided_pen', True),
                'fake_ls_type': np.random.choice(['"random-uniform_0.7_1.2"',
                                                  '"soft_labels_0.1"',
                                                  '"none"']),
                'real_ls_type': np.random.choice(['"random-uniform_0.7_1.2"',
                                                  '"soft_labels_0.1"',
                                                  '"none"']),
                'wrap_absorb': self.config.get('wrap_absorb', False),
                'd_batch_norm': self.config.get('d_batch_norm', False),
                'red_batch_norm': self.config.get('red_batch_norm', False),

                'reward_type': self.config.get('reward_type', 'gail'),

                'red_epochs': self.config.get('red_epochs', 200),
                'red_lr': self.config.get('red_lr', 5e-4),
                'proportion_of_exp_per_red_update': self.config.get(
                    'proportion_of_exp_per_red_update', 1.),

                'rnd_explo': self.config.get('rnd_explo', False),
                'rnd_batch_norm': self.config.get('rnd_batch_norm', True),
                'rnd_lr': self.config.get('rnd_lr', 5e-4),
                'proportion_of_exp_per_rnd_update': self.config.get(
                    'proportion_of_exp_per_rnd_update', 1.),

                'dyn_batch_norm': self.config.get('dyn_batch_norm', True),
                'dyn_lr': self.config.get('dyn_lr', 5e-4),
                'proportion_of_exp_per_dyn_update': self.config.get(
                    'proportion_of_exp_per_dyn_update', 1.),

                'use_purl': self.config.get('use_purl', False),
                'purl_eta': float(self.config.get('purl_eta', 0.25)),
            }
        else:
            # No search, fixed hyper-parameters
            hpmap = {
                'wandb_project': self.wandb_project,
                'uuid': uuid,
                'cuda': self.config['resources']['cuda'],
                'render': False,
                'record': self.config['logging'].get('record', False),
                'task': self.config['meta']['task'],
                'algo': self.config['meta']['algo'],

                # Training
                'num_timesteps': int(float(self.config.get('num_timesteps', 2e7))),
                'eval_steps_per_iter': self.config.get('eval_steps_per_iter', 10),
                'eval_frequency': self.config.get('eval_frequency', 10),

                # Model
                'perception_stack': self.config['perception_stack'],
                'layer_norm': self.config['layer_norm'],
                'shared_value': self.config['shared_value'],

                # Optimization
                'p_lr': float(self.config.get('p_lr', 3e-4)),
                'v_lr': float(self.config.get('v_lr', 1e-3)),
                'lr_schedule': self.config['lr_schedule'],
                'clip_norm': self.config.get('clip_norm', 5.0),

                # Algorithm
                'rollout_len': self.config.get('rollout_len', 2048),
                'optim_epochs_per_iter': self.config.get('optim_epochs_per_iter', 10),
                'batch_size': self.config.get('batch_size', 128),
                'gamma': self.config.get('gamma', 0.995),
                'gae_lambda': self.config.get('gae_lambda', 0.95),
                'eps': self.config.get('eps', 0.2),
                'baseline_scale': float(self.config.get('baseline_scale', 0.5)),
                'p_ent_reg_scale': self.config.get('p_ent_reg_scale', 0.),

                # Adversarial imitation
                'g_steps': self.config.get('g_steps', 3),
                'd_steps': self.config.get('d_steps', 1),
                'd_lr': float(self.config.get('d_lr', 3e-4)),
                'state_only': self.config.get('state_only', False),
                'minimax_only': self.config.get('minimax_only', True),
                'd_ent_reg_scale': self.config.get('d_ent_reg_scale', 0.001),
                'spectral_norm': self.config.get('spectral_norm', True),
                'grad_pen': self.config.get('grad_pen', True),
                'grad_pen_type': self.config.get('grad_pen_type', 'wgan'),
                'one_sided_pen': self.config.get('one_sided_pen', True),
                'fake_ls_type': self.config.get('fake_ls_type', 'none'),
                'real_ls_type': self.config.get('real_ls_type', 'random-uniform_0.7_1.2'),
                'wrap_absorb': self.config.get('wrap_absorb', False),
                'd_batch_norm': self.config.get('d_batch_norm', False),
                'red_batch_norm': self.config.get('red_batch_norm', False),

                'reward_type': self.config.get('reward_type', 'gail'),

                'red_epochs': self.config.get('red_epochs', 200),
                'red_lr': self.config.get('red_lr', 5e-4),
                'proportion_of_exp_per_red_update': self.config.get(
                    'proportion_of_exp_per_red_update', 1.),

                'rnd_explo': self.config.get('rnd_explo', False),
                'rnd_batch_norm': self.config.get('rnd_batch_norm', True),
                'rnd_lr': self.config.get('rnd_lr', 5e-4),
                'proportion_of_exp_per_rnd_update': self.config.get(
                    'proportion_of_exp_per_rnd_update', 1.),

                'dyn_batch_norm': self.config.get('dyn_batch_norm', True),
                'dyn_lr': self.config.get('dyn_lr', 5e-4),
                'proportion_of_exp_per_dyn_update': self.config.get(
                    'proportion_of_exp_per_dyn_update', 1.),

                'use_purl': self.config.get('use_purl', False),
                'purl_eta': float(self.config.get('purl_eta', 0.25)),
            }

        # Duplicate for each environment
        hpmaps = [self.copy_and_add_env(hpmap, env)
                  for env in self.envs]

        if self.need_demos:
            # Duplicate for each number of demos
            hpmaps = [self.copy_and_add_num_demos(hpmap_, num_demos)
                      for hpmap_ in hpmaps
                      for num_demos in self.num_demos]

        # Duplicate for each seed
        hpmaps = [self.copy_and_add_seed(hpmap_, seed)
                  for hpmap_ in hpmaps
                  for seed in range(self.args.num_seeds)]

        # Verify that the correct number of configs have been created
        assert len(hpmaps) == self.args.num_seeds * len(self.envs) * len(self.num_demos)

        return hpmaps

    def unroll_options(self, hpmap):
        """Transform the dictionary of hyperparameters into a string of bash options"""
        indent = 4 * ' '  # choice: indents are defined as 4 spaces
        arguments = ""

        for k, v in hpmap.items():
            if k in self.bool_args:
                if v is False:
                    argument = f"no-{k}"
                else:
                    argument = f"{k}"
            else:
                argument = f"{k}={v}"

            arguments += f"{indent}--{argument} \\\n"

        return arguments

    def create_job_str(self, name, command):
        """Build the batch script that launches a job"""

        # Prepend python command with python binary path
        command = os.path.join(os.environ['CONDA_PREFIX'], "bin", command)

        if self.args.deployment == 'slurm':
            os.makedirs("./out", exist_ok=True)
            # Set sbatch config
            bash_script_str = ('#!/usr/bin/env bash\n\n')
            bash_script_str += (f"#SBATCH --job-name={name}\n"
                                f"#SBATCH --partition={self.partition}\n"
                                f"#SBATCH --ntasks={self.args.num_workers}\n"
                                "#SBATCH --cpus-per-task=1\n"
                                f"#SBATCH --time={self.duration}\n"
                                f"#SBATCH --mem={self.memory}000\n"
                                "#SBATCH --output=./out/run_%j.out\n"
                                '#SBATCH --constraint="V3|V4|V5|V6|V7"\n')  # single quote to escape
            if self.config['resources']['cuda']:
                contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
                bash_script_str += ("#SBATCH --gres=gpu:1\n"
                                    f'#SBATCH --constraint="{contraint}"\n')  # single quote to escape
            bash_script_str += ('\n')
            # Load modules
            bash_script_str += ("module load GCC/8.3.0 OpenMPI/3.1.4\n")
            if self.config['meta']['benchmark'] == 'dmc':  # legacy comment: needed for dmc too
                bash_script_str += ("module load Mesa/19.2.1\n")
            if self.config['resources']['cuda']:
                bash_script_str += ("module load CUDA\n")
            bash_script_str += ('\n')
            # Launch command
            bash_script_str += (f"srun {command}")

        elif self.args.deployment == 'tmux':
            # Set header
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"# job name: {name}\n\n")
            # Launch command
            bash_script_str += (f"mpiexec -n {self.args.num_workers} {command}")

        else:
            raise NotImplementedError("cluster selected is not covered.")

        return bash_script_str[:-2]  # remove the last `\` and `\n` tokens


def run(args):
    """Spawn jobs"""

    if args.wandb_upgrade:
        # Upgrade the wandb package
        logger.info(">>>>>>>>>>>>>>>>>>>> Upgrading wandb pip package")
        out = subprocess.check_output([sys.executable, '-m', 'pip', 'install', 'wandb', '--upgrade'])
        logger.info(out.decode("utf-8"))

    # Create a spawner object
    spawner = Spawner(args)

    # Create directory for spawned jobs
    root = os.path.dirname(os.path.abspath(__file__))
    spawn_dir = os.path.join(root, 'spawn')
    os.makedirs(spawn_dir, exist_ok=True)
    if args.deployment == 'tmux':
        tmux_dir = os.path.join(root, 'tmux')
        os.makedirs(tmux_dir, exist_ok=True)

    # Get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [spawner.get_hps()
                   for _ in range(spawner.config['num_trials'])]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = spawner.get_hps()

    # Create associated task strings
    commands = ["python main.py \\\n{}".format(spawner.unroll_options(hpmap)) for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # Create the job maps
    names = [f"{spawner.type}.{hpmap['uuid']}" for i, hpmap in enumerate(hpmaps)]

    # Finally get all the required job strings
    jobs = [spawner.create_job_str(name, command)
            for name, command in zipsame(names, commands)]

    # Spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(f"job#={i},name={name} -> ready to be deployed.")
        if args.debug:
            logger.info("config below.")
            logger.info(job + "\n")
        dirname = name.split('.')[1]
        full_dirname = os.path.join(spawn_dir, dirname)
        os.makedirs(full_dirname, exist_ok=True)
        job_name = os.path.join(full_dirname, f"{name}.sh")
        with open(job_name, 'w') as f:
            f.write(job)
        if args.deploy_now and not args.deployment == 'tmux':
            # Spawn the job!
            stdout = subprocess.run(["sbatch", job_name]).stdout
            if args.debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f"job#={i},name={name} -> deployed on slurm.")

    if args.deployment == 'tmux':
        dir_ = hpmaps[0]['uuid'].split('.')[0]  # arbitrarilly picked index 0
        session_name = f"{spawner.type}-{str(args.num_seeds).zfill(2)}seeds-{dir_}"
        yaml_content = {'session_name': session_name,
                        'windows': []}
        if spawner.need_demos:
            yaml_content.update({'environment': {'DEMO_DIR': os.environ['DEMO_DIR']}})
        for i, name in enumerate(names):
            executable = f"{name}.sh"
            pane = {'shell_command': [f"source activate {args.conda_env}",
                                      f"chmod u+x spawn/{dir_}/{executable}",
                                      f"spawn/{dir_}/{executable}"]}
            window = {'window_name': f"job{str(i).zfill(2)}",
                      'focus': False,
                      'panes': [pane]}
            yaml_content['windows'].append(window)
            logger.info(f"job#={i},name={name} -> will run in tmux, session={session_name},window={i}.")
        # Dump the assembled tmux config into a yaml file
        job_config = os.path.join(tmux_dir, f"{session_name}.yaml")
        with open(job_config, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        if args.deploy_now:
            # Spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", job_config]).stdout
            if args.debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f"[{len(jobs)}] jobs are now running in tmux session '{session_name}'.")
    else:
        # Summarize the number of jobs spawned
        logger.info(f"[{len(jobs)}] jobs were spawned.")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Job Spawner")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--conda_env', type=str, default=None)
    parser.add_argument('--env_bundle', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--deployment', type=str, choices=['tmux', 'slurm'], default='tmux', help='deploy how?')
    parser.add_argument('--num_seeds', type=int, default=None)
    parser.add_argument('--caliber', type=str, choices=['short', 'long', 'verylong', 'veryverylong'],
                        default='short')
    boolean_flag(parser, 'deploy_now', default=True, help="deploy immediately?")
    boolean_flag(parser, 'sweep', default=False, help="hp search?")
    boolean_flag(parser, 'wandb_upgrade', default=True, help="upgrade wandb?")
    parser.add_argument('--num_demos', '--list', nargs='+', type=str, default=None)
    boolean_flag(parser, 'debug', default=False, help="toggle debug/verbose mode in spawner")
    boolean_flag(parser, 'wandb_dryrun', default=True, help="toggle wandb offline mode")
    parser.add_argument('--debug_lvl', type=int, default=0, help="set the debug level for the spawned runs")
    args = parser.parse_args()

    if args.wandb_dryrun:
        # Run wandb in offline mode (does not sync with wandb servers in real time,
        # use `wandb sync` later on the local directory in `wandb/` to sync to the wandb cloud hosted app)
        os.environ["WANDB_MODE"] = "dryrun"

    # Set the debug level for the spawned runs
    os.environ["DEBUG_LVL"] = str(args.debug_lvl)

    # Create (and optionally deploy) the jobs
    run(args)
