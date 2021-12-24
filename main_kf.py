from jax.config import config
config.update("jax_enable_x64", True)

import pdb
import sys
import argparse
import numpy as np
from pykalman import KalmanFilter

import jax.random as jrandom

from data_generation import gen_slds_nica
from utils import matching_sources_corr

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=3,
                        help="number of ICs")
    parser.add_argument('-m', type=int, default=12,
                        help="dimension of observed data")
    parser.add_argument('-l', type=int, default=1,
                        help="number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('-d', type=int, default=2,
                        help="dimension of lds state")
    parser.add_argument('-k', type=int, default=2,
                        help="number of HMM states")
    parser.add_argument('-t', type=int, default=100000,
                        help="number of timesteps")
    parser.add_argument('--whiten', action='store_true', default=False,
                        help="PCA whiten data as preprocessing")
    # set seeds
    parser.add_argument('--param-seed', type=int, default=50,
                        help="seed for initializing data generation params")
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for initializing data generation sampling")
    parser.add_argument('--est-seed', type=int, default=99,
                        help="seed for initializing function estimators")

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

    # generate nonlinear ICA data
    # !BEWARE d=2, k=2 fixed in datageneration
    x, f, z, z_mu, states, *params = gen_slds_nica(args.n, args.m, args.t,
                                                   args.k, args.d, args.l,
                                                   param_key, data_key,
                                                   repeat_layers=True)

    if args.whiten:
        pca = PCA(whiten=True)
        x = pca.fit_transform(x.T).T

    # Kalman smoothing with EM
    # set initial values
    print("Initializing parameters...")
    np.random.seed(args.est_seed)
    b_prior_est = np.zeros(args.n)+np.random.normal(scale=0.1, size=(args.n,))
    V_prior_est = np.eye(args.n)+np.diag(np.random.normal(scale=0.1,
                                                          size=(args.n,)))
    B_est = np.eye(args.n)+np.diag(np.random.normal(scale=0.1, size=(args.n,)))
    C_est = np.eye(args.m, args.n)+np.random.uniform(low=-0.2, high=0.2,
                                                     size=(args.m, args.n))
    V_est = np.eye(args.n)+np.diag(np.random.normal(scale=0.1, size=(args.n,)))
    R_est = np.eye(args.m)+np.diag(np.random.normal(scale=0.1, size=(args.m,)))

    # transfer to standard numpy 
    measurements = np.array(x.T)
    kf = KalmanFilter(
                      initial_state_mean=b_prior_est,
                      initial_state_covariance=V_prior_est,
                      transition_matrices=B_est,
                      observation_matrices=C_est,
                      transition_covariance=V_est,
                      observation_covariance=R_est,
                      em_vars=['initial_state_mean',
                               'initial_state_covariance',
                               'transition_matrices',
                               'observation_matrices',
                               'transition_covariance',
                               'observation_covariance'],
                      n_dim_obs=args.m,
                      n_dim_state=args.n,
                      random_state=args.est_seed)
    print("Running EM inference...")
    kf_fit = kf.em(measurements, n_iter=5)
    print("..Done!")
    kf_smoothed = kf_fit.smooth(measurements)
    z_mu_est = kf_smoothed[0].T
    C_est = kf_fit.observation_matrices
    f_est = C_est @ z_mu_est

    # evaluate
    logl = kf_fit.loglikelihood(measurements)
    mcc, _, _ = matching_sources_corr(z_mu_est, z_mu[:, :, 0])
    denoise_mcc = np.abs(np.diag(
        np.corrcoef(f_est, f)[:args.m, args.m:])).mean()

    print("*ELBO: {0}\t"
          "mcc: {corr: .2f}\t"
          "denoise mcc: {dcorr: .2f}\t"
          "eseed: {es}\t"
          "pseed: {ps}".format(logl, corr=mcc, dcorr=denoise_mcc,
                               es=args.est_seed, ps=args.param_seed))
    pdb.set_trace()

if __name__ == "__main__":
    sys.exit(main())
