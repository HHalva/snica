import os
import argparse
import sys
import pickle
import pdb

from pathlib import Path
from jax import random as jrandom
from sklearn.decomposition import PCA


from train import train
from subfunc.showdata import *

from sica_data_generation import gen_slds_nica
from sica_load_data import pca

def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=3,
                        help="number of latent components")
    parser.add_argument('-m', type=int, default=12,
                        help="number of latent components")
    parser.add_argument('-t', type=int, default=100000, # 100000
                        help="number of time steps")
    parser.add_argument('-l', type=int, default=1, # 3
                        help="number of mixing layers")
    parser.add_argument('-d', type=int, default=2, # 3
                        help="--not needed here--")
    parser.add_argument('-k', type=int, default=2, # 3
                        help="--state PER n in SICA--")
    parser.add_argument('--whiten', action='store_true', default=False,
                        help="PCA whiten data as preprocessing")

    # set seeds
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for sampling data generation")
    parser.add_argument('--est-seed', type=int, default=0, # 7
                        help="seed for initializing function estimator mlp")
    parser.add_argument('--distrib-seed', type=int, default=0, # 7
                        help="seed for estimating distribution paramaters")
    # training & optimization parameters
    parser.add_argument('--hidden-units', type=int, default=32,
                        help="num. of hidden units in function estimator MLP")
    parser.add_argument('--learning-rate', type=float, default=1e-2, # 3e-4
                        help="learning rate for training")
    parser.add_argument('--num-epochs', type=int, default=100000,
                        help="number of training epochs")
    parser.add_argument('--subseq-len', type=int, default=100,
                        help="length of subsequences")
    parser.add_argument('--minibatch-size', type=int, default=64,
                        help="number of subsequences in a minibatch")
    parser.add_argument('--decay-rate', type=float, default=0.9,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=1000,
                        help="interval (in iterations) for full decay of LR")
    # CUDA settings
    parser.add_argument('--cuda', action='store_true', default=True,
                        help="use GPU training")
    # saving
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")
    args = parser.parse_args()
    return args


def main():
    args = parse()

    train_dir_base = './output/hmmmiia'
    fileid = 'temp.pkl'

    filepath = os.path.join(train_dir_base, fileid)

    # load sica data
    datakey = jrandom.PRNGKey(args.data_seed)
    param_key, data_key = jrandom.split(datakey)

    # generate nonlinear ICA data
    # !BEWARE d=2, k=2 fixed in datageneration
    x, f, z, z_mu, states, *params = gen_slds_nica(args.n, args.m, args.t,
                                                   args.k, args.d, args.l,
                                                   param_key, data_key,
                                                   repeat_layers=True)


    # dimension reduction must be done by PCA
    if args.m > args.n:
        x_data, pca_parm = pca(x, num_comp=args.n)
        x_data = x_data.T
    elif args.m == args.n:
        x_data = x.T
    else:
        raise ValueError('Output dimension must be equal or greater than latent\
                         dimension')
    s = z_mu[:, :, 0]
    s_data = s.T
    # this is only used for some variable shapes later on...
    state_seq = states.T

    # create variable dicts for training
    data_dict = {'x_data': x_data,
                 's_data': s_data,
                 'state_seq': state_seq}
    # note we add one extra layer because the two codes differ in how
    # they define number of layers
    train_dict = {'mix_depth': args.l+1,
                  'hidden_size': args.hidden_units,
                  'learning_rate': args.learning_rate,
                  'num_epochs': args.num_epochs,
                  'subseq_len': args.subseq_len,
                  'minib_size': args.minibatch_size,
                  'decay_rate': args.decay_rate,
                  'decay_steps': args.decay_interval}

    seed_dict = {'est_mlp_seed': args.est_seed,
                 'est_distrib_seed': args.distrib_seed}

    tcl_dict = {'initial_learning_rate': 0.05,
                'momentum': 0.9,
                'max_steps': int(1e5),
                'decay_steps': int(5e4),
                'decay_factor': 0.1,
                'batch_size': 512,
                'moving_average_decay': 0.99, # 0.999
                'checkpoint_steps': int(1e7),
                'summary_steps': int(5e2),
                'weight_decay': 1e-5,
                'train_dir': None,
                'random_seed': seed_dict['est_mlp_seed']}

    # set up dict to save results
    results_dict = {}
    results_dict['data_config'] = {'N': args.n, 'K': args.k**args.n, 'T': args.t,
                                   'mix_depth': args.l,
                                   'data_seed': args.data_seed
                                   }
    results_dict['train_config'] = {'train_vars': train_dict,
                                    'train_seeds': seed_dict}
    results_dict['results'] = []
    results_dict['pca'] = pca
    # results_dict['bests'] = []

    # train HM-nICA model
    s_est, sort_idx, results_dict, est_params = train(
        data_dict, train_dict, seed_dict, tcl_dict, results_dict
    )

    print("Save parameters to %s ..." % filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

    # # save
    # if not os.path.exists(train_dir_base):
    #     Path(train_dir_base).mkdir(parents=True)
    # with open(args.out_dir+"all_results.pickle", 'ab') as out:
    #     pickle.dump(results_dict, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    sys.exit(main())
