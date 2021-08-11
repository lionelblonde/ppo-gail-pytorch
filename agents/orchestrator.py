import time
from copy import deepcopy
import os
import os.path as osp
from collections import defaultdict
import signal

import wandb
import numpy as np

from gym import spaces

from helpers import logger
# from helpers.distributed_util import sync_check
from helpers.env_makers import get_benchmark
from helpers.console_util import timed_cm_wrapper, log_iter_info
from helpers.opencv_util import record_video


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 1)


def ppo_rollout_generator(env, agent, rollout_len):

    benchmark = get_benchmark(agent.hps.env_id)
    t = 0
    done = True
    # Reset agent's env
    ob = np.array(env.reset())
    # Init collections
    rollout = defaultdict(list)
    # Init current episode statistics
    cur_ep_len = 0
    cur_ep_env_ret = 0
    if benchmark == 'safety':
        cur_ep_env_cost = 0

    while True:

        # Predict
        ac, v, logp = agent.predict(ob, sample_or_mode=True)

        if not agent.is_discrete:
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        if t > 0 and t % rollout_len == 0:

            for k in rollout.keys():
                if k in ['obs0', 'obs1']:
                    rollout[k] = np.array(rollout[k]).reshape(-1, *agent.ob_shape)
                elif k == 'acs':
                    rollout[k] = np.array(rollout[k]).reshape(-1, *agent.ac_shape)
                elif k in ['vs', 'logps', 'env_rews', 'env_costs', 'dones']:
                    rollout[k] = np.array(rollout[k]).reshape(-1, 1)
                else:
                    rollout[k] = np.array(rollout[k])
            rollout['next_v'].append(v * (1 - done))

            yield rollout

            # Clear the collections
            rollout.clear()

        # Interact with env(s)
        new_ob, env_rew, done, info = env.step(ac)

        if benchmark == 'safety':
            env_cost = info.get('cost', 0)
            # Add the cost to the reward
            env_rew -= env_cost

        if agent.hps.rnd_explo:
            env_rew += agent.rnd.get_int_rew(new_ob[None])

        # Populate collections
        rollout['obs0'].append(ob)
        rollout['acs'].append(ac)
        rollout['obs1'].append(new_ob)
        rollout['vs'].append(v)
        rollout['logps'].append(logp)
        rollout['env_rews'].append(env_rew)
        if benchmark == 'safety':
            rollout['env_costs'].append(env_cost)
        rollout['dones'].append(done)

        # Update current episode statistics
        cur_ep_len += 1
        # elapsed_steps = (env._elapsed_steps
        #                  if hasattr(env, '_elapsed_steps')
        #                  else env.steps)  # for safety gym
        # assert elapsed_steps == cur_ep_len  # sanity check
        cur_ep_env_ret += env_rew
        if benchmark == 'safety':
            cur_ep_env_cost += env_cost

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # Update the global episodic statistics and
            # reset current episode statistics
            rollout['ep_lens'].append(cur_ep_len)
            cur_ep_len = 0
            rollout['ep_env_rets'].append(cur_ep_env_ret)
            cur_ep_env_ret = 0
            if benchmark == 'safety':
                rollout['ep_env_costs'].append(cur_ep_env_cost)
                cur_ep_env_cost = 0
            # Reset env
            ob = np.array(env.reset())

        t += 1


def gail_rollout_generator(env, agent, rollout_len):

    benchmark = get_benchmark(agent.hps.env_id)
    t = 0
    done = True
    # Reset agent's env
    ob = np.array(env.reset())
    # Init collections
    rollout = defaultdict(list)
    # Init current episode statistics
    cur_ep_len = 0
    cur_ep_env_ret = 0
    cur_ep_syn_ret = 0
    if benchmark == 'safety':
        cur_ep_env_cost = 0

    while True:

        # Predict
        ac, v, logp = agent.predict(ob, sample_or_mode=True)

        if not isinstance(agent.ac_space, spaces.Discrete):
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        if t > 0 and t % rollout_len == 0:

            if agent.hps.wrap_absorb:
                ob_dim = agent.ob_dim + 1
                ac_dim = agent.ac_dim + 1
            else:
                ob_dim = agent.ob_dim
                ac_dim = agent.ac_dim

            for k in rollout.keys():
                if k in ['obs0', 'obs1']:
                    rollout[k] = np.array(rollout[k]).reshape(-1, ob_dim)
                elif k in ['obs0_orig', 'obs1_orig']:
                    rollout[k] = np.array(rollout[k]).reshape(-1, agent.ob_dim)
                elif k == 'acs':
                    rollout[k] = np.array(rollout[k]).reshape(-1, ac_dim)
                elif k in ['vs', 'logps', 'env_rews', 'syn_rews', 'dones']:
                    rollout[k] = np.array(rollout[k]).reshape(-1, 1)
                else:
                    rollout[k] = np.array(rollout[k])
            rollout['next_v'].append(v * (1 - done))

            yield rollout

            # Clear the collections
            rollout.clear()

        # Interact with env(s)
        new_ob, env_rew, done, info = env.step(ac)

        if benchmark == 'safety':
            env_cost = info.get('cost', 0)

        elapsed_steps = (env._elapsed_steps
                         if hasattr(env, '_elapsed_steps')
                         else env.steps)  # for safety gym
        max_episode_steps = (env._max_episode_steps
                             if hasattr(env, '_elapsed_steps')
                             else env.num_steps)  # for safety gym

        # Populate collections
        if agent.hps.wrap_absorb:
            rollout['obs0_orig'].append(ob)
            rollout['acs_orig'].append(ac)
            rollout['obs1_orig'].append(new_ob)
            _ob = np.append(ob, 0)
            _ac = np.append(ac, 0)
            rollout['obs0'].append(_ob)
            rollout['acs'].append(_ac)
            rollout['vs'].append(v)
            rollout['logps'].append(logp)
            rollout['env_rews'].append(env_rew)
            if benchmark == 'safety':
                rollout['env_costs'].append(env_cost)
            rollout['dones'].append(done)
            if done and not elapsed_steps == max_episode_steps:
                # Wrap with an absorbing state
                _new_ob = np.append(np.zeros(agent.ob_shape), 1)
                rollout['obs1'].append(_new_ob)
                syn_rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                if agent.hps.rnd_explo:
                    syn_rew += agent.rnd.get_int_rew(_new_ob[None])
                syn_rew = np.asscalar(syn_rew.detach().cpu().numpy().flatten())
                rollout['syn_rews'].append(syn_rew)
                # Add absorbing transition
                rollout['obs0_orig'].append(ob)
                rollout['acs_orig'].append(ac)
                rollout['obs1_orig'].append(new_ob)
                rollout['obs0'].append(np.append(np.zeros(agent.ob_shape), 1))
                rollout['acs'].append(np.append(np.zeros(agent.ac_shape), 1))
                rollout['obs1'].append(np.append(np.zeros(agent.ob_shape), 1))
                rollout['vs'].append(v)
                rollout['logps'].append(logp)
                rollout['env_rews'].append(env_rew)
                if benchmark == 'safety':
                    rollout['env_costs'].append(env_cost)
                rollout['dones'].append(done)
                rollout['syn_rews'].append(syn_rew)
            else:
                _new_ob = np.append(new_ob, 0)
                rollout['obs1'].append(_new_ob)
                syn_rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                if agent.hps.rnd_explo:
                    syn_rew += agent.rnd.get_int_rew(_new_ob[None])
                syn_rew = np.asscalar(syn_rew.detach().cpu().numpy().flatten())
                rollout['syn_rews'].append(syn_rew)
        else:
            rollout['obs0'].append(ob)
            rollout['acs'].append(ac)
            rollout['obs1'].append(new_ob)
            rollout['vs'].append(v)
            rollout['logps'].append(logp)
            rollout['env_rews'].append(env_rew)
            if benchmark == 'safety':
                rollout['env_costs'].append(env_cost)
            rollout['dones'].append(done)
            syn_rew = agent.get_syn_rew(ob[None], ac[None], new_ob[None])
            if agent.hps.rnd_explo:
                syn_rew += agent.rnd.get_int_rew(_new_ob[None])
            syn_rew = np.asscalar(syn_rew.detach().cpu().numpy().flatten())
            rollout['syn_rews'].append(syn_rew)

        # Update current episode statistics
        cur_ep_len += 1
        assert elapsed_steps == cur_ep_len  # sanity check
        cur_ep_env_ret += env_rew
        if benchmark == 'safety':
            cur_ep_env_cost += env_cost
        cur_ep_syn_ret += syn_rew

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # Update the global episodic statistics and
            # reset current episode statistics
            rollout['ep_lens'].append(cur_ep_len)
            cur_ep_len = 0
            rollout['ep_env_rets'].append(cur_ep_env_ret)
            cur_ep_env_ret = 0
            rollout['ep_syn_rets'].append(cur_ep_syn_ret)
            cur_ep_syn_ret = 0
            if benchmark == 'safety':
                rollout['ep_env_costs'].append(cur_ep_env_cost)
                cur_ep_env_cost = 0
            # Reset env
            ob = np.array(env.reset())

        t += 1


def ep_generator(env, agent, render, record):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    benchmark = get_benchmark(agent.hps.env_id)

    if record:

        width = 320 if benchmark == 'dmc' else 500
        height = 240 if benchmark == 'dmc' else 500

        def bgr_to_rgb(x):
            _b = np.expand_dims(x[..., 0], -1)
            _g = np.expand_dims(x[..., 1], -1)
            _r = np.expand_dims(x[..., 2], -1)
            rgb_x = np.concatenate([_r, _g, _b], axis=-1)
            del x, _b, _g, _r
            return rgb_x

        kwargs = {'mode': 'rgb_array'}
        if benchmark in ['mujoco', 'atari']:
            def _render():
                return bgr_to_rgb(env.render(**kwargs))
        elif benchmark in ['dmc', 'safety']:
            def _render():
                x = deepcopy(env.render(mode='rgb_array',
                                        camera_id=1,
                                        width=width,
                                        height=height))
                _b = np.expand_dims(x[..., 0], -1)
                _g = np.expand_dims(x[..., 1], -1)
                _r = np.expand_dims(x[..., 2], -1)
                rgb_x = np.concatenate([_r, _g, _b], axis=-1)
                del x, _b, _g, _r
                return rgb_x
        elif benchmark in ['pycolab']:
            def _render():
                return env.render(**kwargs)
        else:
            raise ValueError('unsupported benchmark')

    ob = np.array(env.reset())

    if record:
        ob_orig = _render()

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    if record:
        obs_render = []
    acs = []
    vs = []
    env_rews = []
    if benchmark == 'safety':
        cur_ep_env_cost = 0
        env_costs = []

    while True:
        ac, v, _ = agent.predict(ob, sample_or_mode=False)

        if not agent.is_discrete:
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else np.asscalar(ac)

        obs.append(ob)
        if record:
            obs_render.append(ob_orig)
        acs.append(ac)
        vs.append(v)
        new_ob, env_rew, done, info = env.step(ac)

        if render:
            env.render()

        if record:
            ob_orig = _render()

        if benchmark == 'safety':
            env_cost = info.get('cost', 0)

        env_rews.append(env_rew)
        if benchmark == 'safety':
            env_costs.append(env_cost)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        if benchmark == 'safety':
            cur_ep_env_cost += env_cost
        ob = np.array(deepcopy(new_ob))
        if done:
            obs = np.array(obs)
            if record:
                obs_render = np.array(obs_render)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {"obs": obs,
                   "acs": acs,
                   "vs": vs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            if record:
                out.update({"obs_render": obs_render})
            if benchmark == 'safety':
                out.update({"env_costs": env_costs,
                            "ep_env_cost": cur_ep_env_cost})

            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            if record:
                obs_render = []
            acs = []
            env_rews = []
            if benchmark == 'safety':
                cur_ep_env_cost = 0
                env_costs = []

            ob = np.array(env.reset())

            if record:
                ob_orig = _render()


def evaluate(args,
             env,
             agent_wrapper,
             experiment_name):

    # Create an agent
    agent = agent_wrapper()

    # Create episode generator
    ep_gen = ep_generator(env, agent, args.render, args.record)

    if args.record:
        vid_dir = osp.join(args.video_dir, experiment_name)
        os.makedirs(vid_dir, exist_ok=True)

    # Load the model
    agent.load(args.model_path, args.iter_num)
    logger.info("model loaded from path:\n  {}".format(args.model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(args.num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, args.num_trajs))
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
        if args.record:
            # Record a video of the episode
            record_video(vid_dir, i, traj['obs_render'])

    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(args,
          rank,
          env,
          eval_env,
          agent_wrapper,
          experiment_name):

    # Create an agent
    agent = agent_wrapper()

    # Get benchmark
    benchmark = get_benchmark(agent.hps.env_id)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger, use=DEBUG)

    num_iters = args.num_timesteps // args.rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()

    if rank == 0:
        # Create collections
        d = defaultdict(list)

        # Set up model save directory
        ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save the model as a dry run, to avoid bad surprises at the end
        agent.save(ckpt_dir, "{}_dryrun".format(iters_so_far))
        logger.info("dry run. Saving model @: {}".format(ckpt_dir))
        if args.record:
            vid_dir = osp.join(args.video_dir, experiment_name)
            os.makedirs(vid_dir, exist_ok=True)

        # Handle timeout signal gracefully
        def timeout(signum, frame):
            # Save the model
            agent.save(ckpt_dir, "{}_timeout".format(iters_so_far))
            # No need to log a message, orterun stopped the trace already
            # No need to end the run by hand, SIGKILL is sent by orterun fast enough after SIGTERM

        # Tie the timeout handler with the termination signal
        # Note, orterun relays SIGTERM and SIGINT to the workers as SIGTERM signals,
        # quickly followed by a SIGKILL signal (Open-MPI impl)
        signal.signal(signal.SIGTERM, timeout)

        # Group by everything except the seed, which is last, hence index -1
        # For 'gail', it groups by uuid + gitSHA + env_id + num_demos,
        # while for 'ppo', it groups by uuid + gitSHA + env_id
        group = '.'.join(experiment_name.split('.')[:-1])

        # Set up wandb
        while True:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=experiment_name,
                    id=experiment_name,
                    group=group,
                    config=args.__dict__,
                    dir=args.root,
                )
                break
            except Exception:
                pause = 10
                logger.info("wandb co error. Retrying in {} secs.".format(pause))
                time.sleep(pause)
        logger.info("wandb co established!")

    # Create rollout generator for training the agent
    if args.algo.split('_')[0] == 'ppo':
        roll_gen = ppo_rollout_generator(env, agent, args.rollout_len)
    elif args.algo.split('_')[0] == 'gail':
        roll_gen = gail_rollout_generator(env, agent, args.rollout_len)
    else:
        raise NotImplementedError
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, args.render, args.record)

    while iters_so_far <= num_iters:

        if iters_so_far % 100 == 0 or DEBUG:
            log_iter_info(logger, iters_so_far, num_iters, tstart)

        # if iters_so_far % 20 == 0:
        #     # Check if the mpi workers are still synced
        #     sync_check(agent.policy)
        #     if agent.hps.algo.split('_')[0] == 'gail':
        #         sync_check(agent.discriminator)

        if args.algo.split('_')[0] == 'ppo':

            with timed('interacting'):
                rollout = roll_gen.__next__()

            with timed('training'):
                metrics, lrnow = agent.update_policy_value(
                    rollout=rollout,
                    iters_so_far=iters_so_far,
                )
                if rank == 0 and iters_so_far % args.eval_frequency == 0:
                    # Log training stats
                    d['pol_losses'].append(metrics['p_loss'])
                    d['val_losses'].append(metrics['v_loss'])

        elif args.algo.split('_')[0] == 'gail':

            for _ in range(agent.hps.g_steps):
                with timed('interacting'):
                    rollout = roll_gen.__next__()

                with timed('policy and value training'):
                    metrics, lrnow = agent.update_policy_value(
                        rollout=rollout,
                        iters_so_far=iters_so_far,
                    )
                    if rank == 0 and iters_so_far % args.eval_frequency == 0:
                        # Log training stats
                        d['pol_losses'].append(metrics['p_loss'])
                        d['val_losses'].append(metrics['v_loss'])

            for _ in range(agent.hps.d_steps):
                with timed('discriminator training'):
                    metrics = agent.update_discriminator(
                        rollout=rollout,
                    )
                    if rank == 0 and iters_so_far % args.eval_frequency == 0:
                        # Log training stats
                        d['dis_losses'].append(metrics['d_loss'])

        if rank == 0 and iters_so_far % args.eval_frequency == 0:

            with timed("evaluating"):

                for eval_step in range(args.eval_steps_per_iter):
                    # Sample an episode w/ non-perturbed actor w/o storing anything
                    eval_ep = eval_ep_gen.__next__()
                    # Aggregate data collected during the evaluation to the buffers
                    d['eval_len'].append(eval_ep['ep_len'])
                    d['eval_env_ret'].append(eval_ep['ep_env_ret'])
                    if benchmark == 'safety':
                        d['eval_env_cost'].append(eval_ep['ep_env_cost'])

        # Increment counters
        iters_so_far += 1
        if args.algo.split('_')[0] == 'ppo':
            timesteps_so_far += args.rollout_len
        elif args.algo.split('_')[0] == 'gail':
            timesteps_so_far += (agent.hps.g_steps * args.rollout_len)

        if rank == 0 and (iters_so_far - 1) % args.eval_frequency == 0:

            # Log stats in csv
            logger.record_tabular('timestep', timesteps_so_far)
            logger.record_tabular('eval_len', np.mean(d['eval_len']))
            logger.record_tabular('eval_env_ret', np.mean(d['eval_env_ret']))
            if benchmark == 'safety':
                logger.record_tabular('eval_env_cost', np.mean(d['eval_env_cost']))
            logger.info("dumping stats in .csv file")
            logger.dump_tabular()

            if args.record:
                # Record the last episode in a video
                record_video(vid_dir, iters_so_far, eval_ep['obs_render'])

            # Log stats in dashboard
            wandb.log({'pol_loss': np.mean(d['pol_losses']),
                       'val_loss': np.mean(d['val_losses']),
                       'lrnow': np.array(lrnow)},
                      step=timesteps_so_far)

            if args.algo.split('_')[0] == 'gail':
                wandb.log({'dis_loss': np.mean(d['dis_losses'])},
                          step=timesteps_so_far)

            wandb.log({'eval_len': np.mean(d['eval_len']),
                       'eval_env_ret': np.mean(d['eval_env_ret'])},
                      step=timesteps_so_far)
            if benchmark == 'safety':
                wandb.log({'eval_env_cost': np.mean(d['eval_env_cost'])},
                          step=timesteps_so_far)

            # Clear the iteration's running stats
            d.clear()

    if rank == 0:
        # Save once we are done iterating
        agent.save(ckpt_dir, iters_so_far)
        logger.info("we're done. Saving model @: {}".format(ckpt_dir))
        logger.info("bye.")
