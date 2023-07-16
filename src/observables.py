import numpy as np
from src.physics import commutator, np_expval, np_anticommutator


def calc_eignevalues(data: np.ndarray):
    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")

    evals = np.linalg.eigvals(data)
    return np.sort(evals)


def entropy_vn(rho: np.ndarray, base=np.e, fast=True):
    vals = calc_eignevalues(rho)
    nzvals = vals[vals != 0]  # Not-zero eigenvalues
    if base == 2:
        log = np.log2 if fast else np.lib.scimath.log2
    elif base == np.e:
        log = np.log if fast else np.lib.scimath.log
    else:
        raise ValueError("Base must be 2 or e.")
    logvals = log(nzvals)
    return float(np.real(-np.sum(nzvals * logvals)))


def purity(rho: np.ndarray):
    return np.trace(rho @ rho)


def cov_matrix_element(rho, op1, op2):
    comm = np_anticommutator(op1, op2)
    return 0.5 * np_expval(rho, comm) - np_expval(rho, op1) * np_expval(rho, op2)


def covariance(rho: np.ndarray, operators):
    cov = [
        [cov_matrix_element(rho, operators[k], operators[l])
         for k in range(len(operators))]
        for l in range(len(operators))
    ]
    return cov
