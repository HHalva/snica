import argparse
import pdb
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from jax import config

#import jax.random as jrandom
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

#from data_generation import gen_slds_nica
#from train import full_train

# uncomment to debug NaNs
#config.update("jax_debug_nans", True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # load configs
    cfg = cfg.experiments

    # get data
    if cfg.experiment_name == 'snica_synthetic':
        pdb.set_trace()
        x, f, z, z_mu, states, *params = gen_slds_nica(cfg.data_gen)
    #else:
    #    sys.exit()

    ## train
    #train(x, f, z_mu, states, cfg)


if __name__ == "__main__":
    sys.exit(main())
