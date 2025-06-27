import argparse
import collections
import hydra
import logging
from numpyro import optim
from omegaconf import DictConfig
import os
import rootutils
from typing import Any, Callable, Dict, List, Optional, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.data.datamodule import DataModule
from src.trainer import ParaMonad, Trainer
from src.utils import extras, get_metric_value, task_wrapper

log = logging.LoggerAdapter(logger=logging.getLogger(__name__))

@task_wrapper
def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating generative model <{cfg.model._target_}>")
    model: Callable = hydra.utils.instantiate(cfg.model)
    log.info(f"Instantiating guide inference program <{cfg.guide._target_}>")
    guide: Callable = hydra.utils.instantiate(cfg.guide)

    log.info(f"Instantiating trainable module <{cfg.monad._target_}>")
    monad: ParaMonad = hydra.utils.instantiate(cfg.monad,
                                               data_shape=datamodule.shape,
                                               guide=guide, model=model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: BaseTrainer = hydra.utils.instantiate(cfg.trainer, logger=log)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "monad": monad,
        "trainer": trainer,
    }

    log.info("Starting testing!")
    if cfg.ckpt_path == "" or not os.path.exists(cfg.ckpt_path):
        log.warning("Best ckpt not found! Using current weights for testing...")
        cfg.ckpt_path = None
    test_metrics = trainer.test(monad, datamodule, ckpt_path=cfg.ckpt_path,
                                valid=False)
    log.info(f"Tested from ckpt path: {cfg.ckpt_path}")

    # merge train and test metrics
    metric_dict = {**test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for testing.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = test(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
