"""
This script tests out different versions of NMF to see what
works best to deconstruct the matrix such that the reconstruction
error on the test set is highest.
"""

import json
import numpy as np
import argparse
import joblib

from sklearn.decomposition import NMF
from sklearn.feature_selection import r_regression


def main(target_dir, alpha_H=0, alpha_W=0, l1_ratio=0, tol=1e-3, init="nndsvdar"):
    """
    Implement the NMF algorithm and save the best results
    """

    # Print out the arguments
    print("target_dir: ", target_dir)
    print("alpha_H: ", alpha_H)
    print("alpha_W: ", alpha_W)
    print("l1_ratio: ", l1_ratio)
    print("tol: ", tol)
    print("init: ", init)
    # Load the data
    X, y = load_data("train.json")

    # Implement the NMF algorithm, using the best parameters found.
    nmf = NMF(
        n_components=500,
        init="nndsvdar",
        alpha_H=alpha_H,
        alpha_W=alpha_W,
        l1_ratio=l1_ratio,
        tol=tol,
        random_state=0,
        verbose=True,
    )
    s_nmf = nmf.fit_transform(X)

    # Print the fit
    print("NMF fit: ", nmf.reconstruction_err_)
    # Find the value that correlates most with y
    corr = r_regression(s_nmf, y)
    pred_order = np.argsort(corr)[::-1]

    W_nmf = nmf.components_[pred_order]

    # Save the results
    joblib.dump(W_nmf, target_dir + "W_nmf.joblib")
    joblib.dump(nmf, target_dir + "nmf.joblib")


def load_data(filename):
    """
    Load the data and aggregate the matrices into
    a single matrix.
    """
    input_dir = "/hpc/group/carlsonlab/zdc6/wildfire/data/count_vectorized/"
    train = joblib.load(input_dir + "train.joblib")

    X = []
    y = []
    for day in train:
        X.append(day["aqi"])
        y.append(day["tweets"])

    return np.array(X), np.array(y)


if __name__ == "__main__":

    nmf_dir = "/hpc/group/carlsonlab/zdc6/wildfire/data/nmf/"

    parser = argparse.ArgumentParser(description="Calculate the NMF")

    parser.add_argument("-ah", type=float, default=0.0, help="alpha_H")
    parser.add_argument("-aw", type=float, default=0.0, help="alpha_W")
    parser.add_argument("-i", type=str, default="nndsvdar", help="init")
    parser.add_argument("-l1", type=float, default=0.0, help="l1_ratio")
    parser.add_argument("-t", type=str, default=nmf_dir, help="target directory")
    parser.add_argument("-tol", type=float, default=1e-3, help="tolerance")

    args = parser.parse_args()

    main(
        args.ah,
        args.aw,
        args.l1,
        args.t,
        args.tol,
        args.i,
    )
