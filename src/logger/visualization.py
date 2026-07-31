import importlib
from datetime import datetime
import numpy as np
from omegaconf import DictConfig
from typing import Any, Dict, Optional


def _scalar(value):
    """Coerce a Jax/Numpy scalar into something a logging backend accepts."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, TypeError, ValueError):
            pass
    return value


class TensorboardWriter():
    def __init__(self, log_dir, logger=None, enabled=True):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                if logger is not None:
                    logger.warning(message)
                else:
                    print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Record the (already flattened) hyperparameters of the run as text.
        """
        if self.writer is None:
            return
        lines = "\n".join("%s: %s" % (k, params[k]) for k in sorted(params))
        self.writer.add_text('hparams', "```\n" + lines + "\n```", 0)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int]=None,
                    step_metric: Optional[str]=None):
        """
        Record summary metrics under their verbatim tags, i.e. without the
        train/valid mode suffix that `add_scalar` appends.
        """
        if self.writer is None:
            return
        step = self.step if step is None else step
        for k, v in metrics.items():
            if k != step_metric:
                self.writer.add_scalar(k, _scalar(v), step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class WandbWriter():
    """
    Weights & Biases counterpart to `TensorboardWriter`, exposing the same
    `set_step`/`add_*` interface so that the two are interchangeable (and
    composable, see `CompositeWriter`).

    Scalars are buffered until the next `set_step` call, so that everything
    recorded for one training step lands in a single `wandb.log` row.  Since
    training and validation steps are not globally monotonic (each restarts
    per-epoch counting from its own dataloader length), the step is logged as a
    per-mode metric (`train/step`, `valid/step`) which W&B is told to use as the
    x-axis, rather than as W&B's own internal step.
    """

    tb_writer_ftns = {
        'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
        'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
    }
    tag_mode_exceptions = {'add_histogram', 'add_embedding'}

    def __init__(self, project: str="numpyro-template",
                 api_key: Optional[str]=None, entity: Optional[str]=None,
                 group: Optional[str]=None, host: Optional[str]=None,
                 id: Optional[str]=None, job_type: Optional[str]=None,
                 name: Optional[str]=None, notes: Optional[str]=None,
                 resume: Optional[str]=None, save_dir: Optional[str]=None,
                 tags=None, mode: str="online", enabled: bool=True,
                 log_images: bool=True, max_images: int=16, num_bins: int=64,
                 logger=None):
        self.run = None
        self.wandb = None
        self.log_images = log_images
        self.max_images = max_images
        self.num_bins = num_bins
        self.step = 0
        self.mode = ''
        self.timer = datetime.now()
        self._buffer = {}
        self._step_metrics = {}

        if not enabled:
            return

        try:
            import wandb
        except ImportError:
            message = "Warning: visualization (Weights & Biases) is configured "\
                      "to use, but wandb is not installed on this machine. "\
                      "Please install it with 'pip install wandb' or select "\
                      "another logger with 'logger=tensorboard'."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
            return

        self.wandb = wandb
        if api_key is not None:
            wandb.login(key=api_key, host=host, relogin=True)
        self.run = wandb.init(
            dir=save_dir, entity=entity, group=group, id=id, job_type=job_type,
            mode=mode, name=name, notes=notes, project=project, resume=resume,
            settings=wandb.Settings(base_url=host) if host else None,
            tags=list(tags) if tags is not None else None,
        )

    def set_step(self, step, mode='train'):
        self._flush()
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def add_scalar(self, tag, data, *args, **kwargs):
        self._record('add_scalar', tag, _scalar(data))

    def add_scalars(self, tag, data, *args, **kwargs):
        for k, v in data.items():
            self._record('add_scalars', '{}/{}'.format(tag, k), _scalar(v))

    def add_image(self, tag, data, *args, **kwargs):
        if self.log_images:
            self._record('add_image', tag, self._image(data))

    def add_images(self, tag, data, *args, **kwargs):
        if self.log_images:
            images = np.asarray(data)[:self.max_images]
            self._record('add_images', tag, [self._image(i) for i in images])

    def add_histogram(self, tag, data, *args, **kwargs):
        if self.run is None:
            return
        values = np.asarray(data).flatten()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self._record('add_histogram', tag,
                     self.wandb.Histogram(values, num_bins=self.num_bins))

    def add_text(self, tag, data, *args, **kwargs):
        self._record('add_text', tag, str(data))

    def log_hyperparams(self, params: Dict[str, Any]):
        if self.run is not None:
            self.run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int]=None,
                    step_metric: Optional[str]=None):
        """
        Record summary metrics under their verbatim tags, i.e. without the
        train/valid mode suffix that `add_scalar` appends.  When `step_metric`
        names one of the metrics (e.g. "epoch"), W&B plots the others against
        it.
        """
        if self.run is None:
            return
        self._flush()
        payload = {k: _scalar(v) for k, v in metrics.items()}
        if step is not None:
            payload.setdefault('step', step)
        if step_metric is not None and step_metric in payload:
            for k in payload:
                if k != step_metric:
                    self._define_metric(k, step_metric)
        self.run.log(payload)

    def close(self):
        self._flush()
        if self.run is not None:
            self.run.finish()
            self.run = None

    def _define_metric(self, tag, step_metric):
        """
        Tell W&B to plot `tag` against `step_metric` instead of its own
        internal step counter.
        """
        if self._step_metrics.get(tag) == step_metric:
            return
        self._step_metrics[tag] = step_metric
        if step_metric not in self._step_metrics:
            self._step_metrics[step_metric] = None
            self.run.define_metric(step_metric, hidden=True)
        self.run.define_metric(tag, step_metric=step_metric)

    def _flush(self):
        if self.run is not None and self._buffer:
            self.run.log(self._buffer)
        self._buffer = {}

    def _image(self, image):
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)  # CHW -> HWC
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        return self.wandb.Image(image)

    def _record(self, name, tag, value):
        if self.run is None:
            return
        if self.mode and name not in self.tag_mode_exceptions:
            tag = '{}/{}'.format(tag, self.mode)
        self._buffer[tag] = value
        if self.mode:
            step_metric = '{}/step'.format(self.mode)
            self._buffer[step_metric] = self.step
            self._define_metric(tag, step_metric)

    def __getattr__(self, name):
        """
        Silently ignore the `add_*` methods of the Tensorboard interface that
        have no Weights & Biases counterpart here.
        """
        if name in self.tb_writer_ftns:
            return lambda tag, data, *args, **kwargs: None
        raise AttributeError("type object 'WandbWriter' has no attribute "
                             "'{}'".format(name))


class CompositeWriter():
    """
    Fans every writer call out to a collection of writers, so that a run can log
    to several backends (say Tensorboard and Weights & Biases) at once.
    """

    def __init__(self, *writers):
        self.writers = [writer for writer in writers if writer is not None]

    def __iter__(self):
        return iter(self.writers)

    def __len__(self):
        return len(self.writers)

    def __getattr__(self, name):
        if not name.startswith(('add_', 'log_', 'set_')) and name != 'close':
            raise AttributeError("type object 'CompositeWriter' has no "
                                 "attribute '{}'".format(name))

        def wrapper(*args, **kwargs):
            for writer in self.writers:
                getattr(writer, name)(*args, **kwargs)
        return wrapper


def instantiate_writers(cfg: Optional[DictConfig], logger=None):
    """
    Instantiate every writer configured under the `logger` config group.

    :param cfg: DictConfig mapping writer names to their instantiable configs.
    :param logger: Python logger handed to each writer for its own warnings.
    :return: CompositeWriter over the instantiated writers.
    """
    import hydra

    if not cfg:
        message = "No logger configs found! Skipping visualization writers..."
        if logger is not None:
            logger.warning(message)
        else:
            print(message)
        return CompositeWriter()

    if not isinstance(cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    writers = []
    for conf in cfg.values():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            if logger is not None:
                logger.info(f"Instantiating logger <{conf._target_}>")
            writers.append(hydra.utils.instantiate(conf, logger=logger))
    return CompositeWriter(*writers)
