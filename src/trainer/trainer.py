from functools import partial
import itertools
from jax import jit, random
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
from numpyro.infer import SVI, Trace_ELBO
from base import BaseTrainer
from utils import flatten, inf_loop, MetricTracker

@jit
def binarize(rng_key, batch):
    return random.bernoulli(rng_key, batch).astype(batch.dtype)

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, callables, rng_seed, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 num_particles=4):
        super().__init__(callables, [], optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(jnp.sqrt(data_loader.batch_size))
        self.num_particles = num_particles
        self.svi = SVI(self.callables.model, self.callables.guide,
                       self.optimizer, Trace_ELBO())

        self.rng, rng_svi = random.split(PRNGKey(rng_seed), 2)
        self.svi = SVI(self.callables.model, self.callables.guide,
                       self.optimizer, Trace_ELBO())
        for (data, _, _) in self.data_loader:
            init_batch = data
            break
        self.svi_state = self.svi.init(rng_svi, binarize(self.rng, init_batch))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.train_metrics.reset()
        for batch_idx, (data, target, idx) in enumerate(self.data_loader):
            self.rng, rng_binarize = random.split(random.fold_in(self.rng, batch_idx))
            data = binarize(rng_binarize, data)
            self.svi_state, loss = self._train_step(self.svi_state, data)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    @partial(jit, static_argnums=0)
    def _train_step(self, state, data):
        return self.svi.update(state, data)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.valid_metrics.reset()
        for batch_idx, (data, target, idx) in enumerate(self.valid_data_loader):
            self.rng, rng_binarize = random.split(random.fold_in(self.rng, batch_idx))
            data = binarize(rng_binarize, data)
            loss = self.svi.evaluate(self.svi_state, data)

            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(target))

        # # add histogram of model and guide parameters to the tensorboard
        parameters = self.optimizer.get_params(self.svi_state.optim_state)
        for name in parameters:
            for p, par in enumerate(flatten(parameters[name])):
                self.writer.add_histogram(name + "$" + str(p), np.asarray(par),
                                          bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
