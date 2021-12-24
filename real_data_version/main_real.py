import os
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/appl/opt/CUDA/10.2.89-GCC-8.3.0"
#os.environ["MPLCONFIGDIR"]  = "/wrk/users/herhal"
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import argparse
import pickle
from pathlib import Path

import pdb

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, value_and_grad
from jax.lax import scan
from jax.ops import index, index_add
from jax.experimental import optimizers, host_callback
from jax.config import config
config.update("jax_enable_x64", True)

from train_real import train
from elbo import ELBO
from utils import matching_sources_corr

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from jax.config import config
#config.update("jax_debug_nans", True)


def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=5,
                        help="number of ICs")
    parser.add_argument('-d', type=int, default=2,
                        help="dimension of lds state")
    parser.add_argument('-k', type=int, default=2,
                        help="number of HMM states")
    parser.add_argument('--prob-stay', type=float, default=0.99,
                        help="probability of staying in a HMM state")
    parser.add_argument('--whiten', action='store_true', default=False,
                        help="PCA whiten data as preprocessing")
    # set seeds
    parser.add_argument('--est-seed', type=int, default=0,
                        help="seed for initializing function estimators")
    # arguments to initialize graphical model parameters
    parser.add_argument('--init-xprec-scale', type=int, default=100,
                        help="seed for initializing function estimators")
    # inference & training & optimization parameters
    parser.add_argument('--inference-iters', type=int, default=30,
                        help="num. of inference iterations")
    parser.add_argument('--num-samples', type=int, default=20,
                        help="num. of samples for elbo")
    parser.add_argument('--hidden-units-enc', type=int, default=32,
                        help="num. of hidden units in encoder estimator MLP")
    parser.add_argument('--hidden-units-dec', type=int, default=16,
                        help="num. of hidden units in decoder estimator MLP")
    parser.add_argument('--hidden-layers-enc', type=int, default=4,
                        help="num. of hidden layers in encoder estimator MLP")
    parser.add_argument('--hidden-layers-dec', type=int, default=2,
                        help="num. of hidden layers in decoder estimator MLP")
    parser.add_argument('--nn-learning-rate', type=float, default=3e-3,
                        help="learning rate for training function estimators")
    parser.add_argument('--gm-learning-rate', type=float, default=3e-6,
                        help="learning rate for training GM parameters")
    parser.add_argument('--num-epochs', type=int, default=1,
                        help="number of training epochs")
    parser.add_argument('--decay-rate', type=float, default=1.,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=1000,
                        help="interval (in iterations) for full decay of LR")
    parser.add_argument('--subseq-len', type=int, default=200,
                        help="T is split into this length sub-chains")
    parser.add_argument('--minib-size', type=int, default=64,
                        help="number of subchains in a single minibatch")
    parser.add_argument('--vmp', action='store_true', default=False,
                        help="use VMP rather than SVAE")
    parser.add_argument('--print-freq', type=int, default=10,
                        help="printin elbo frequency")

    # saving
    parser.add_argument('--save-freq', type=str, default=10,
                        help="save checkpoint every _ epoch")
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where output is saved")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    # load data
    with open(r"meg_data_sample.pkl", "rb") as input_file:
        x = pickle.load(input_file)

    # as an example, split into train and test
    x_tr, x_te = jnp.split(x, 2, axis=1)

    # option for whitening with PCA (have not tested this at all so may not
    # work)
    if args.whiten:
        pca = PCA(whiten=True)
        x = pca.fit_transform(x.T).T

    # train
    (phi, theta), (R_est, lds_est, hmm_est), best_elbo = train(x_tr, args)

    # compute posteriors by performing infernce of full test set
    evalkey = jrandom.PRNGKey(999999)
    print("Now running infernce on full test data for evaluation")
    elbo, (qz, qzlag_z, qu, quu) = ELBO(x_te, R_est, lds_est, hmm_est, phi,
             theta, evalkey, args.inference_iters, args.num_samples)
    # get posterior means of ICs
    qs = qz[0][:, :, 0]

    # as example compute MCC with itself
    mcc, _, _ = matching_sources_corr(qs, qs)
    print("ELBO on test set:", elbo, "MCC:", mcc)

    pdb.set_trace()

    # save more info -- not done yet
    #fileid = vars(args).copy()
    #rm_from_id = ('cuda', 'out_dir', 'print_freq')
    #for k in rm_from_id:
    #    fileid.pop(k, None)
    #fileid = [str(i)+str(j) for i, j in zip(fileid.keys(), fileid.values())]
    #fileid = '-'.join(fileid)
    #out_dict = dict()
    #out_dict['args'] = vars(args)
    #out_dict['x'] = x
    #out_dict['est_params'] = est_params
    #out_dict['posteriors'] = posteriors
    #out_dict['best'] = best_dict

    #if not os.path.exists(args.out_dir):
    #    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    #onp.savez_compressed(args.out_dir+fileid, **out_dict)


if __name__ == "__main__":
    main()
