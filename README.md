# PyTorch implementations of PPO and GAIL

The repository covers the following algorithms:
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) (Reinforcement Learning):
    - MLP and CNN policies
    - Gaussian policies (continuous actions) and categorical policies (discrete actions)
    - Benchmarks and environments covered (installation instructions below):
        - MuJoCo (OpenAI Gym)
        - DeepMind Control Suite
        - Safety Gym
        - Atari (OpenAI Gym)
        - Pycolab
- [Generative Adversarial Imitation Learning (GAIL)](http://arxiv.org/abs/1606.03476) (Imitation Learning):
    - MLP policies
    - Gaussian policies (continuous actions)
    - Benchmarks and environments covered (installation instructions below):
        - MuJoCo (OpenAI Gym)
    - Expert demonstrations to mimic
    are made available at
    [this link](https://drive.google.com/drive/folders/1dGw-O6ZT_WWTuqDayIA9xat1jZgeiXoE?usp=sharing)
    (installation instructions below).

The repository also offers implementations of the following:
- [Random Network Distillation (RND)](http://arxiv.org/abs/1810.12894) exploration (intrinsic motivation-based) reward bonus
- [Random Expert Distillation (RED)](http://arxiv.org/abs/1905.06750) imitation reward (based on RND)
- Model-based imitation reward modulation, penalizing the agent when its *controlability* is lower (metric aligned
with the prediction error of the learned forward model, like *curiosity*, but in the opposite direction)
- Absorbing state wrapping mechanism as described in [Discriminator-Actor-Critic (DAC)](http://arxiv.org/abs/1809.02925)
- [Positive-Unlabeled Reward Learning (PURL)](http://arxiv.org/abs/1911.00459) discriminator training

## Dependencies

### OS

Make sure you have [GLFW](https://www.glfw.org) and [Open MPI](https://www.open-mpi.org) installed on your system:
- if you are using macOS, run:
```bash
brew install open-mpi glfw3
```
- if you are using Ubuntu, run:
```bash
sudo apt -y install libopenmpi-dev libglfw3
```

### Python

Create a virtual enviroment for Python development using
[Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda)
or [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda):
- Create a conda environment for Python 3.7 called 'myenv', activate it, and upgrade `pip`:
```bash
conda create -n myenv python=3.7
conda activate myenv
# Once in the conda environment, upgrade the pip binary it uses to the latest
pip install --upgrade pip
```
- Install various core Python libraries:
```bash
# EITHER with versions that were used for this release
pip install pytest==5.2.1 pytest-instafail==0.4.1 flake8==3.7.9 wrapt==1.11.2 pillow==6.2.1 six==1.15.0 tqdm==4.36.1 pyyaml==5.1.2 psutil==5.6.3 cloudpickle==1.2.2 tmuxp==1.5.4 lockfile==0.12.2 numpy==1.17.4 pandas==0.25.2 scipy==1.3.1 scikit-learn==0.21.3 h5py==2.10.0 matplotlib==3.1.1 seaborn==0.9.0 pyvips==2.1.8 scikit-image==0.16.2 torch==1.6.0 torchvision==0.7.0
conda install -y -c conda-forge opencv=3.4.7 pyglet=1.3.2 pyopengl=3.1.5 mpi4py=3.0.2 cython=0.29.13 watchdog=0.9.0
pip install moviepy==1.0.1 imageio==2.6.1 wandb==0.10.10
# OR without versions (pulls the latest versions for each of these releases)
pip install pytest pytest-instafail flake8 wrapt pillow six tqdm pyyaml psutil cloudpickle tmuxp lockfile numpy pandas scipy scikit-learn h5py matplotlib seaborn pyvips scikit-image torch torchvision
conda install -y -c conda-forge opencv pyglet pyopengl mpi4py cython watchdog
pip install moviepy imageio wandb
```
- Install MuJoCo following the instructions laid out at
[`https://github.com/openai/mujoco-py#install-mujoco`](https://github.com/openai/mujoco-py#install-mujoco)
- Build [`mujoco-py`](https://github.com/openai/mujoco-py.git) from source in editable mode:
```bash
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
```
- Build [`gym`](https://github.com/openai/gym.git) from source in editable mode
(*cf.* [`http://gym.openai.com/docs/#installation`](http://gym.openai.com/docs/#installation)):
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e ".[all]"
```
- [Optional] Build [`safety-gym`](https://github.com/openai/safety-gym) from source in editable mode
(*cf.* [`https://github.com/openai/safety-gym#installation`](https://github.com/openai/safety-gym#installation)):
```bash
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
```
- [Optional] Build [`pycolab`](https://github.com/deepmind/pycolab) from source in editable mode:
```bash
git clone https://github.com/deepmind/pycolab
cd pycolab
pip install -e .
```
- [Optional] Build `dm_control` from source in editable mode
(*cf.* [`https://github.com/deepmind/dm_control#requirements-and-installation`](https://github.com/deepmind/dm_control#requirements-and-installation)):
```bash
git clone https://github.com/deepmind/dm_control
cd dm_control
pip install -e .
```

### Expert Demonstrations

Download the expert demonstrations complementing this repository and make them accessible:
- Download the expert demonstrations that we have shared at
[this link](https://drive.google.com/drive/folders/1dGw-O6ZT_WWTuqDayIA9xat1jZgeiXoE?usp=sharing);
- Place them at the desired location in your filesystem;
- Create the environment variable: `export DEMO_DIR=/where/you/downloaded/and/placed/the/demos`.

## Running Experiments
While one can launch any job via `main.py`, it is advised to use `spawner.py`,
designed to spawn a swarm of experiments over multiple seeds and environments in one command.
To get its usage description, type `python spawner.py -h`.
```bash
usage: spawner.py [-h] [--config CONFIG] [--conda_env CONDA_ENV]
                  [--env_bundle ENV_BUNDLE] [--num_workers NUM_WORKERS]
                  [--deployment {tmux,slurm}] [--num_seeds NUM_SEEDS]
                  [--caliber {short,long,verylong,veryverylong}]
                  [--deploy_now] [--no-deploy_now] [--sweep] [--no-sweep]
                  [--wandb_upgrade] [--no-wandb_upgrade]
                  [--num_demos NUM_DEMOS [NUM_DEMOS ...]] [--debug]
                  [--no-debug] [--wandb_dryrun] [--no-wandb_dryrun]
                  [--debug_lvl DEBUG_LVL]

Job Spawner

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG
  --conda_env CONDA_ENV
  --env_bundle ENV_BUNDLE
  --num_workers NUM_WORKERS
  --deployment {tmux,slurm}
                        deploy how?
  --num_seeds NUM_SEEDS
  --caliber {short,long,verylong,veryverylong}
  --deploy_now          deploy immediately?
  --no-deploy_now
  --sweep               hp search?
  --no-sweep
  --wandb_upgrade       upgrade wandb?
  --no-wandb_upgrade
  --num_demos NUM_DEMOS [NUM_DEMOS ...], --list NUM_DEMOS [NUM_DEMOS ...]
  --debug               toggle debug/verbose mode in spawner
  --no-debug
  --wandb_dryrun        toggle wandb offline mode
  --no-wandb_dryrun
  --debug_lvl DEBUG_LVL
                        set the debug level for the spawned runs
```

Here is an example:
```bash
python spawner.py --config tasks/train_mujoco_ppo.yaml --env_bundle debug --wandb_upgrade --no-sweep --deploy_now --caliber short --num_workers 2 --num_seeds 3 --deployment tmux --conda_env myenv --wandb_dryrun --debug_lvl 2
```
Check the argument parser in `spawner.py` to know what each of these arguments mean,
and how to adapt them to your needs.
