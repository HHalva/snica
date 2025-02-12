import argparse
import pdb
import sys
import hydra
from omegaconf import DictConfig

from jax import config

#import jax.random as jrandom
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

from data_generation import gen_slds_nica
from train import train

# uncomment to debug NaNs
#config.update("jax_debug_nans", True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # load configs
    cfg = cfg.experiments

    if cfg.experiment_name == 'snica_synthetic':
        x, f, z, z_mu, states, *params = gen_slds_nica(cfg.data_gen)
        train(x, cfg)
    #else:
    #    sys.exit()

    # train
    train(x, cfg)


if __name__ == "__main__":
    sys.exit(main())
