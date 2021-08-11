import numpy as np


def gae(segment, gamma, lam, rew_key):
    """'Generalized Advantage Estimation' (GAE)
    Schulman, ICLR 2016, https://arxiv.org/abs/1506.02438

    Args:
        segment (dict): Collected segment of transitions (>< episode)
        gamma (float): Discount factor, same letter in the GAE paper
        lam (float): GAE parameter, 'lambda' in the GAE paper
        rew_key (str): Key associated with the reward entity
    """
    # Extract segment length
    length = len(segment[rew_key])
    # Augment the predicted values with the last predicted one (last not included in segment)
    # If the last `done` is 1, `segment['next_v']` is 0 (see segment generator)
    vs = np.append(segment["vs"], segment["next_v"])
    # Mark the last transition as nonterminal dones[length+1] = 0
    # If this is wrong and length+1 is in fact terminal (<=> dones[length+1] = 1),
    # then vs[length+1] = 0 by definition (see segment generator), so when we later
    # treat t = length in the loop, setting dones[length+1] = 0 leads to the correct values,
    # whether length+1 is terminal or non-terminal.
    dones = np.append(segment["dones"], 0)

    # Create empty GAE-modded advantage vector (same length as the reward minibatch)
    rews = segment[rew_key]
    gae_advs = np.empty_like(rews, dtype='float32')

    last_gae_adv = 0
    # Using a reversed loop naturally stacks the powers of gamma * lambda by wrapping
    # e.g. gae_rews[T-3] = delta[T-2] +
    #                      gamma * lambda * (delta[T-1] + gamma * lambda * delta[T])
    # = delta[T-2] + gamma * lambda * delta[T-1] + (gamma * lambda)**2 * delta[T]
    # The computed GAE advantage relies only on deltas of future timesteps, hence the reversed
    for t in reversed(range(length)):
        # Whether the current transition is terminal
        nonterminal = 1 - dones[t + 1]
        # Compute the 1-step Temporal Difference residual
        delta = rews[t] + (gamma * vs[t + 1] * nonterminal) - vs[t]
        # Compute the GAE-modded advantage and add it to the advantage vector
        last_gae_adv = delta + (gamma * lam * nonterminal * last_gae_adv)
        gae_advs[t] = last_gae_adv

    # Augment the segment with the constructed statistics
    segment["advs"] = gae_advs  # vector containing the GAE advantages
    # Add the values (baselines) to the advantages to get the returns (MC Q)
    segment["td_lam_rets"] = gae_advs + segment["vs"]
