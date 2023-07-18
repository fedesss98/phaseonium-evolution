import numpy as np
from src.physics import commutator, np_expval, np_anticommutator


def calc_eignevalues(data: np.ndarray):
    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")

    evals = np.linalg.eigvals(data)
    return np.sort(evals)


def entropy_vn(rho: np.ndarray, base=np.e, fast=True):
    """ Tr[rho log(rho)] done by diagonalizing the density matrix"""
    vals = calc_eignevalues(rho)
    nzvals = vals[vals != 0]  # Not-zero eigenvalues
    # Choose function to use
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
    return np.array(cov)


def covariance_invariants(covariance_matrix: np.ndarray):
    """Ferraro, Olivares, Paris - 2005; pag.21"""
    a = covariance_matrix[0:2, 0:2]
    b = covariance_matrix[2:4, 2:4]
    c = covariance_matrix[0:2, 2:4]
    i1 = np.linalg.det(a)
    i2 = np.linalg.det(b)
    i3 = np.linalg.det(c)
    i4 = np.linalg.det(covariance_matrix)
    return [i1, i2, i3, i4]


def symplectic_eigenvalues(covariance_matrix: np.ndarray):
    """Ferraro, Olivares, Paris - 2005; pag.22"""
    i1, i2, i3, i4 = covariance_invariants(covariance_matrix)
    first_factor = i1 + i2 + 2 * i3
    second_factor = np.sqrt(first_factor ** 2 - 4 * i4)
    d1 = np.sqrt((first_factor + second_factor) / 2)
    d2 = np.sqrt((first_factor - second_factor) / 2)
    return d1, d2


def minientropy(d):
    """Ferraro, Olivares, Paris - 2005; pag.22"""
    x1 = d + 1/2
    x2 = d - 1/2
    return x1*np.log(x1) - x2*np.log(x2)


def symplectic_entropy(nus):
    """Ferraro, Olivares, Paris - 2005; pag.22"""
    return sum([minientropy(nu) for nu in nus if nu != 0])


def symplectic_purity(nus):
    return np.prod([1/nu for nu in nus])
