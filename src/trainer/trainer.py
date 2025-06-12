from abc import abstractmethod, abstractproperty
from collections import defaultdict
from functools import partial
from jax.random import PRNGKey
import math
import numpy as np
from numpyro.infer import SVI
import os
from rich.progress import track
from typing import Callable, List, Optional

from src.data import DataModule
from src.logger import TensorboardWriter
from src.utils import flatten, inf_loop, MetricTracker
from .para import ParaMonad

def _progress(batch_idx, data_loader):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len(data_loader)
    return base.format(current, total, 100.0 * current / total)

class Trainer:
    """
    Base class for all trainers
    """
    def __init__(self, check_val_every_n_epoch: int=1, default_root_dir="",
                 early_stop: Optional[int]=None, epochs: int=1, logger=None,
                 log_step: int=10, metrics: List[str]=[],
                 monitor: Optional[str]=None, min_epochs: int=1,
                 resume: Optional[str]=None, save_period: int=1,
                 tensorboard: bool=True, validate: bool=True):
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.checkpoint_dir = default_root_dir + "/saved/"
        self.default_root_dir = default_root_dir
        self.early_stop = np.inf if early_stop is None else early_stop
        self.epochs = epochs
        self.logger = logger
        self.log_dir = default_root_dir
        self.log_step = log_step
        self.metrics = ["loss"] + metrics
        self.min_epochs = min_epochs
        self.monitor = "off" if monitor is None else monitor
        self.save_period = save_period
        self.validate = validate

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = np.inf if self.monitor_mode == 'min' else -np.inf

        self.epoch = 0

        # setup visualization writer instance
        self.writer = TensorboardWriter(self.log_dir, self.logger, tensorboard)

        self.train_metrics = MetricTracker(*self.metrics, prefix="train",
                                           writer=self.writer)
        self.valid_metrics = MetricTracker(*self.metrics, prefix="valid",
                                           writer=self.writer)

    @abstractproperty
    def metric_fns(self) -> List[str]:
        raise NotImplementedError

    def _resume_checkpoint(self, monad, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {}.npy ...".format(resume_path))
        checkpoint = np.load(resume_path)
        self.epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        try:
            monad.load(checkpoint)
        except Exception as ex:
            self.logger.exception(ex.msg)
            raise ex

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.epoch))

    def _save_checkpoint(self, monad, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'monitor_best': self.monitor_best,
            **monad.save()
        }
        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}'.format(epoch))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        np.save(filename, state, allow_pickle=True)
        self.logger.info("Saving checkpoint: {}.npy ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir + '/model_best')
            np.save(best_path, state, allow_pickle=True)
            self.logger.info("Saving current best: model_best ...")

    def _train_epoch(self, monad, data_loader, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.log_step is None:
            log_step = int(math.sqrt(data_loader.batch_size))
        else:
            log_step = self.log_step
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(data_loader):
            metrics = monad.train_step(*batch)
            loss = metrics['loss'].item()

            self.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
            for met in self.metrics:
                self.train_metrics.update(met, metrics[met])

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, _progress(batch_idx, data_loader), loss
                ))

        return self.train_metrics.result()

    def test(self, monad: ParaMonad, datamodule: DataModule,
             ckpt_path: Optional[str]=None, valid: bool=True):
        dataloader = datamodule.valid_dataloader() if valid else\
                     datamodule.test_dataloader()
        metrics = defaultdict(lambda: [])
        for batch_idx, batch in enumerate(dataloader):
            for k, v in monad.valid_step(*batch).items():
                metrics[k].append(v)
        return {k: np.mean(vs) for k, vs in metrics.items()}

    def train(self, monad: ParaMonad, datamodule: DataModule,
              ckpt_path: Optional[str]=None):
        """
        Full training logic
        """
        not_improved_count = 0
        train_dataloader = datamodule.train_dataloader()
        valid_dataloader = datamodule.valid_dataloader()
        for epoch in range(self.epoch, self.epochs + 1):
            train_result = self._train_epoch(monad, train_dataloader, epoch)
            valid_result = {}
            if self.validate:
                valid_result = self._valid_epoch(monad, valid_dataloader, epoch)

            # save logged information into log dict
            log = {'epoch': epoch}
            log.update(**{'train/'+k : v.item() for k, v in train_result.items()})
            log.update(**{'val/'+k : v.item() for k, v in valid_result.items()})

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(monitor_metric)
                    improved = (self.monitor_mode == 'min' and log[self.monitor_metric] <= self.monitor_best) or \
                               (self.monitor_mode == 'max' and log[self.monitor_metric] >= self.monitor_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.monitor_metric))
                    self.monitor_mode = 'off'
                    improved = False

                if improved:
                    self.monitor_best = log[self.monitor_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(monad, epoch, save_best=best)

    def _valid_epoch(self, monad, data_loader, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.valid_metrics.reset()
        for batch_idx, batch in enumerate(data_loader):
            metrics = monad.valid_step(*batch)
            loss = metrics['loss'].item()

            self.writer.set_step((epoch - 1) * len(data_loader) + batch_idx,
                                 'valid')
            for met in self.metrics:
                self.valid_metrics.update(met, metrics[met])

        # add histogram of parameters to the tensorboard
        parameters = monad.parameters
        for name in parameters:
            for p, par in enumerate(flatten(parameters[name])):
                self.writer.add_histogram(name + "$" + str(p), np.asarray(par),
                                          bins='auto')
        return self.valid_metrics.result()
