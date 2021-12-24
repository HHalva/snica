# for remote server only
import os
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/appl/opt/CUDA/10.2.89-GCC-8.3.0"
#os.environ["MPLCONFIGDIR"]  = "/wrk/users/herhal"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import argparse
import numpy as onp
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

from data_generation import gen_slds_linear_ica, gen_slds_nica
from func_estimators import nica_mlp
from train_artificial import train
from full_train_artificial import full_train
from elbo import ELBO
from utils import matching_sources_corr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from jax.config import config
#config.update("jax_debug_nans", True)


def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=6,
                        help="number of ICs")
    parser.add_argument('-m', type=int, default=24,
                        help="dimension of observed data")
    parser.add_argument('-l', type=int, default=0,
                        help="number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('-d', type=int, default=2,
                        help="dimension of lds state")
    parser.add_argument('-k', type=int, default=2,
                        help="number of HMM states")
    parser.add_argument('-t', type=int, default=100000,
                        help="number of timesteps")
    #parser.add_argument('--prob-stay', type=float, default=0.95,
    #                    help="probability of staying in a state")
    parser.add_argument('--whiten', action='store_true', default=False,
                        help="PCA whiten data as preprocessing")
    # set seeds
    parser.add_argument('--param-seed', type=int, default=50,
                        help="seed for initializing data generation params")
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for initializing data generation sampling")
    parser.add_argument('--est-seed', type=int, default=99,
                        help="seed for initializing function estimators")
    parser.add_argument('--eval-seed', type=int, default=666,
                        help="seed for initializing evaluation")
    # inference & training & optimization parameters
    parser.add_argument('--learn-all', action='store_true', default=True,
                        help="learn all parameters")
    parser.add_argument('--inference-iters', type=int, default=30,
                        help="num. of inference iterations")
    parser.add_argument('--num-samples', type=int, default=20,
                        help="num. of samples for elbo")
    parser.add_argument('--hidden-units-enc', type=int, default=64,
                        help="num. of hidden units in encoder estimator MLP")
    parser.add_argument('--hidden-units-dec', type=int, default=64,
                        help="num. of hidden units in decoder estimator MLP")
    parser.add_argument('--hidden-layers-enc', type=int, default=0,
                        help="num. of hidden layers in encoder estimator MLP")
    parser.add_argument('--hidden-layers-dec', type=int, default=0,
                        help="num. of hidden layers in decoder estimator MLP")
    parser.add_argument('--nn-learning-rate', type=float, default=1e-2,
                        help="learning rate for training function estimators")
    parser.add_argument('--gm-learning-rate', type=float, default=1e-2,
                        help="learning rate for training GM parameters")
    parser.add_argument('--dropout', action='store_true', default=False,
                        help="learn all parameters")
    parser.add_argument('--burnin', type=float, default=0,
                        help="keep output precision fixed for _ iterations")
    parser.add_argument('--num-epochs', type=int, default=100000,
                        help="number of training epochs")
    parser.add_argument('--decay-rate', type=float, default=0.9,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=1000,
                        help="interval (in iterations) for full decay of LR")
    parser.add_argument('--subseq-len', type=int, default=100,
                        help="T is split into this length sub-chains")
    parser.add_argument('--minib-size', type=int, default=64,
                        help="number of subchains in a single minibatch")
    parser.add_argument('--vmp', action='store_true', default=False,
                        help="use VMP rather than SVAE")
    parser.add_argument('--eval-freq', type=int, default=40,
                        help="evaluation frequency")
    parser.add_argument('--print-freq', type=int, default=32,
                        help="printing frequency")

    # saving
    parser.add_argument('--save-freq', type=str, default=10,
                        help="save checkpoint every _ epoch")
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")
    args = parser.parse_args()
    return args


def main():
    args = parse()

    # generate data
    param_key = jrandom.PRNGKey(args.param_seed)
    data_key = jrandom.PRNGKey(args.data_seed)
    if args.l == 0:
        # generate linear ICA data
        # !BEWARE d=2 fixed in datageneration
        var_ratio = 0.
        while not(0.08 <= var_ratio <= 0.15):
            print("Generating data with good obs noise ratio...")
            x, z, z_mu, states, *params, var_ratio = gen_slds_linear_ica(
                args.n, args.m, args.t, args.k, args.d, param_key, data_key)
            param_key, _ = jrandom.split(param_key)
            data_key, _ = jrandom.split(data_key)
    elif args.l > 0:
        # generate nonlinear ICA data
        # !BEWARE d=2 fixed in datageneration
        var_ratio = 0.
        while not (0.08 <= var_ratio <= 0.15):
            print("Generating data with good obs noise ratio...")
            x, z, z_mu, states, *params, var_ratio = gen_slds_nica(
                args.n, args.m, args.t, args.k, args.d, args.l, param_key, data_key)
            param_key, _ = jrandom.split(param_key)
            data_key, _ = jrandom.split(data_key)

    # GT decoders
    if args.l == 0:
        C_gt = params[0][0]
    elif args.l > 0:
        dec_gt = params[0][0]

    # load trained models
    name_dict = {"n": args.n, "m": args.m,
                 "l": args.l, "es": args.est_seed,
                 "ps": args.param_seed,
                 "nLR": args.nn_learning_rate,
                 "gLR": args.gm_learning_rate,
                 "uenc": args.hidden_units_enc,
                 "udec": args.hidden_units_enc,
                 "lenc": args.hidden_layers_enc,
                 "ldec": args.hidden_layers_dec,
                 "bin": args.burnin}
    file_id = [str(i)+str(j) for i,j in zip(name_dict.keys(),
                                            name_dict.values())]
    file_id = "_".join(file_id)
    encdec_filename = file_id+"_encdec_ckpt.pkl"
    gm_filename = file_id+"_gm_ckpt.pkl"

    encdec_ckpt = pickle.load(open(
        os.path.join(args.out_dir, encdec_filename), "rb"))
    gm_ckpt = pickle.load(open(os.path.join(args.out_dir, gm_filename), "rb"))
    opt_state_nn = optimizers.pack_optimizer_state(encdec_ckpt)
    opt_state_gm = optimizers.pack_optimizer_state(gm_ckpt)

    # set optimizers (not needed apart from collecting parameters)
    opt_init_nn, opt_update_nn, get_params_nn = optimizers.adam(
        args.nn_learning_rate)
    opt_init_gm, opt_update_gm, get_params_gm = optimizers.adam(
        args.gm_learning_rate)

    # perform inference
    evkey = jrandom.PRNGKey(args.eval_seed)
    phi_ev, theta_ev = get_params_nn(opt_state_nn)
    R_ev, lds_ev, hmm_ev = get_params_gm(opt_state_gm)

    # compute elbo with
    nu = 1 # no annealing
    mode = 1 # ensures dropout is OFF
    elbo, (qz, qzlag_z, qu, quu) = ELBO(x, R_ev, lds_ev, hmm_ev,
                                        phi_ev, theta_ev, nu,
                                        evkey, args.inference_iters,
                                        args.num_samples,
                                        mode)

    mcc_s, s_est, sort_idx = matching_sources_corr(qz[0][:, :, 0],
                                                 z_mu[:, :, 0],
                                                 method='pearson')

    # perform denoising experiment with GT decoder
    s_est = qz[0][:, :, 0]
    s_gt = z[:, :, 0]
    if args.l == 0:
        f_est = C_gt.T @ s_est
        f_gt = C_gt.T @ s_gt
    if args.l > 0:
        f_est = vmap(nica_mlp, (None, 1))(dec_gt, s_est)
        f_gt = vmap(nica_mlp, (None, 1))(dec_gt, s_gt)

    mcc, _, sort_idx = matching_sources_corr(f_est, f_gt)
    r2 = mcc**2

    print("ELBO:", elbo, "IC mcc:", mcc_s, "denoise R2:", r2)

    pdb.set_trace()

if __name__ == "__main__":
    main()
