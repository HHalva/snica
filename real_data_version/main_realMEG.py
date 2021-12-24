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

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from jax.config import config
#config.update("jax_debug_nans", True)
from load_data import load_train_data, load_test_data, downstearm_svm

def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')
    # MEG
    parser.add_argument('--fs', type=float, default=80,
                        help="downsampling fs")
    # data generation args
    parser.add_argument('-m', type=int, default=64,
                        help="num of observed x (pca reduction)")
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
    parser.add_argument('--evalkey', type=float, default=10,
                        help="key of evaluation")

    # saving
    parser.add_argument('--save-freq', type=str, default=10,
                        help="save checkpoint every _ epoch")
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where output is saved")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    # load training data
    x_train, _, sub_idx = load_train_data(num_subs=10, pca_dim=args.m, low_cutoff=4, high_cutoff=30, down_fs=args.fs)

    M, T =x_train.shape
    # train
    (phi, theta), (R_est, lds_est, hmm_est), best_elbo = train(x_train, args)
    results_dict = {'phi':phi, 'theta':theta,'Rest':R_est,'lds':lds_est,'hmm':hmm_est,'elbo':best_elbo}

    # load testing data
    data_list, labels_list = load_test_data(sub_idx, task='passive', low_cutoff=4, high_cutoff=30, down_fs=args.fs)
    
    # downstrem classfication task
    args.evalkey = jrandom.PRNGKey(999999)
    print("Now running infernce on full test data for evaluation")
    test_acc = downstearm_svm(data_list, labels_list, results_dict, args)
    print('Classification accuracy: %.4f' % test_acc)
    
    # save inference params and args
    results_dict = {'phi':phi, 'theta':theta,'Rest':R_est,'lds':lds_est,'hmm':hmm_est,
                    'elbo':best_elbo, 'args':args}
    
    fileid = 'Acc%d_Layer%d_Comp%d_Node%d_Lr%d_Batch%d_SubLen%d_PCA%d.pkl' % (int(test_acc*100),
                                                                                args.hidden_layers_enc,
                                                                                args.n,
                                                                                args.hidden_units_enc,
                                                                                int(args.nn_learning_rate*1000),
                                                                                args.minib_size,
                                                                                args.subseq_len,
                                                                                args.m)
    filepath = os.path.join(args.out_dir, fileid)
    print("Save parameters to %s ..." % filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f, pickle.DEFAULT_PROTOCOL)
#    elbo, (qz, qzlag_z, qu, quu) = ELBO(x_te, R_est, lds_est, hmm_est, phi,
#             theta, evalkey, args.inference_iters, args.num_samples)
#    # get posterior means of ICs
#    qs = qz[0][:, :, 0]
#
#    # as example compute MCC with itself
#    mcc, _, _ = matching_sources_corr(qs, qs)
#    print("ELBO on test set:", elbo, "MCC:", mcc)
#
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
