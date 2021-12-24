import argparse
import pdb
import sys

from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as jrandom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_generation import gen_slds_nica
from train import full_train

# uncomment to debug NaNs
#config.update("jax_debug_nans", True)


def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=3,
                        help="number of ICs")
    parser.add_argument('-m', type=int, default=12,
                        help="dimension of observed data")
    parser.add_argument('-t', type=int, default=100000,
                        help="number of timesteps")
    parser.add_argument('-l', type=int, default=0,
                        help="number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('-d', type=int, default=2,
                        help="dimension of lds state. Fixed at 2 in experim.")
    parser.add_argument('-k', type=int, default=2,
                        help="number of HMM states. Fixed at 2 in experients")
    parser.add_argument('--whiten', action='store_true', default=False,
                        help="PCA whiten data as preprocessing")
    parser.add_argument('--gt-gm-params', action='store_true', default=False,
                        help="debug with GM parameters at ground truth")
    # set seeds
    parser.add_argument('--param-seed', type=int, default=50,
                        help="seed for initializing data generation params")
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for initializing data generation sampling")
    parser.add_argument('--est-seed', type=int, default=99,
                        help="seed for initializing function estimators")
    # inference & training & optimization parameters
    parser.add_argument('--inference-iters', type=int, default=5,
                        help="num. of inference iterations")
    parser.add_argument('--num-samples', type=int, default=1,
                        help="num. of samples for elbo")
    parser.add_argument('--hidden-units-enc', type=int, default=128,
                        help="num. of hidden units in encoder estimator MLP")
    parser.add_argument('--hidden-units-dec', type=int, default=64,
                        help="num. of hidden units in decoder estimator MLP")
    parser.add_argument('--hidden-layers-enc', type=int, default=2,
                        help="num. of hidden layers in encoder estimator MLP")
    parser.add_argument('--hidden-layers-dec', type=int, default=1,
                        help="num. of hidden layers in decoder estimator MLP")
    parser.add_argument('--nn-learning-rate', type=float, default=1e-2,
                        help="learning rate for training function estimators")
    parser.add_argument('--gm-learning-rate', type=float, default=1e-2,
                        help="learning rate for training GM parameters")
    parser.add_argument('--burnin', type=float, default=500,
                        help="keep output precision fixed for _ iterations")
    parser.add_argument('--num-epochs', type=int, default=100000,
                        help="number of training epochs")
    parser.add_argument('--decay-rate', type=float, default=1.,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=1e10,
                        help="interval (in iterations) for full decay of LR")
    parser.add_argument('--plot-freq', type=int, default=10,
                        help="plotting frequency")
    # saving and loading
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")
    parser.add_argument('--resume-best', action='store_true', default=False,
                        help="resume from best chkpoint for current args")
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help="eval only wihtout training")


    args = parser.parse_args()
    return args


def main():
    args = parse()

    # generate data
    param_key = jrandom.PRNGKey(args.param_seed)
    data_key = jrandom.PRNGKey(args.data_seed)

    # generate simulated data
    # !BEWARE d=2, k=2 fixed in data generation code
    x, f, z, z_mu, states, *params = gen_slds_nica(args.n, args.m, args.t,
                                                   args.k, args.d, args.l,
                                                   param_key, data_key,
                                                   repeat_layers=True)

    # we have not tried this option but could be useful in some cases
    if args.whiten:
        pca = PCA(whiten=True)
        x = pca.fit_transform(x.T).T

    # train
    est_params, posteriors, best_elbo = full_train(x, f, z, z_mu, states,
                                                   params, args, args.est_seed)


if __name__ == "__main__":
    sys.exit(main())
