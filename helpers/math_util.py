import scipy.signal


def discount(x, gamma):
    """Compute discounted sum along the 0-th dimension of the `x` ndarray
    Return an ndarray `y` with the same shape as x, satisfying:
        y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ... + gamma^k * x[t+k],
            where k = len(x) - t - 1

    Args:
        x (np.ndarray): 2-D array of floats, time x features
        gamma (float): Discount factor
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class LRScheduler(object):

    def __init__(self, optimizer, initial_lr, lr_schedule, total_num_steps, kwargs=None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr_schedule = lr_schedule
        self.total_num_steps = total_num_steps

    def step(self, steps_so_far):
        """Return the next lr in accordance with the initial lr,
        the desired schedule, and the learning progress.
        """
        if self.lr_schedule == 'linear':
            next_lr = self.initial_lr * max(1.0 - (float(steps_so_far) / float(self.total_num_steps)), 0.)
        elif self.lr_schedule == 'constant':
            next_lr = self.initial_lr * 1.0
        else:
            raise NotImplementedError('invalid lr schedule.')
        for g in self.optimizer.param_groups:
            g['lr'] = next_lr
        return next_lr
