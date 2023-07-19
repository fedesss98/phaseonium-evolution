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
    alpha = covariance_matrix[0:2, 0:2]
    beta = covariance_matrix[2:4, 2:4]
    gamma = covariance_matrix[0:2, 2:4]
    i1 = np.linalg.det(alpha)
    i2 = np.linalg.det(beta)
    i3 = np.linalg.det(gamma)
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


def _get_symplectic_eigenvalues(cov_or_nus: np.ndarray):
    if isinstance(cov_or_nus, np.ndarray) and cov_or_nus.shape == (2,):
        d1, d2 = cov_or_nus
    elif isinstance(cov_or_nus, np.ndarray) and cov_or_nus.shape == (4, 4):
        d1, d2 = symplectic_eigenvalues(cov_or_nus)
    else:
        print(f'Error in covariance matrix: shape is {input.shape} while 4x4 is expected')
        raise ValueError
    return d1, d2


def minientropy(d):
    """Ferraro, Olivares, Paris - 2005; pag.22"""
    x1 = d + 1/2
    x2 = d - 1/2
    return x1*np.log(x1) - x2*np.log(x2)


def symplectic_entropy(cov_or_nus: np.ndarray):
    """Ferraro, Olivares, Paris - 2005; pag.22"""
    d1, d2 = _get_symplectic_eigenvalues(cov_or_nus)
    f = minientropy
    return f(d1) + f(d2)


def symplectic_purity(cov_or_nus: np.ndarray):
    d1, d2 = _get_symplectic_eigenvalues(cov_or_nus)
    return 1 / (d1 * d2)


def _determinant_minimum(a, b, c, d):
    left_hand = (d - a*b) ** 2
    right_hand = (1 + b) * c**2 * (a + d)
    if left_hand <= right_hand:
        e_min = 2*c**2 + (-1+b)*(-a+d) + 2*abs(c) * np.sqrt(c**2 + (-1+b)*(-a+d))
        e_min /= (-1+b) ** 2
    else:
        e_min = a*b - c**2 + d - np.sqrt((c**4 + (-a*b+d)**2 - 2*c**2*(a*b+d)))
        e_min /= 2*b
    return e_min


def gaussian_quantum_discord(covariance_matrix):
    """
    @article{PhysRevLett.105.030501,
      title = {Quantum versus Classical Correlations in Gaussian States},
      author = {Adesso, Gerardo and Datta, Animesh},
      journal = {Phys. Rev. Lett.},
      volume = {105},
      issue = {3},
      pages = {030501},
      numpages = {4},
      year = {2010},
      month = {Jul},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevLett.105.030501},
      url = {https://link.aps.org/doi/10.1103/PhysRevLett.105.030501}
    }
    &
    Su, X. (2013).
    Applying Gaussian quantum discord to quantum key distribution.
    ArXiv. /abs/1310.4253
    """
    invariants = covariance_invariants(covariance_matrix)
    d1, d2 = symplectic_eigenvalues(covariance_matrix)
    e_min = _determinant_minimum(*invariants)
    f = minientropy
    return f(np.sqrt(invariants[1])) - f(d1) - f(d2) - f(np.sqrt(e_min))


def mutual_information(covariance_matrix):
    """
    Serafini, A., Illuminati, F., & De Siena, S. (2003).
    Symplectic invariants, entropic measures and correlations of Gaussian states.
    ArXiv. https://doi.org/10.1088/0953-4075/37/2/L02
    """
    invariants = covariance_invariants(covariance_matrix)
    d1, d2 = symplectic_eigenvalues(covariance_matrix)
    a = np.sqrt(invariants[0])
    b = np.sqrt(invariants[1])
    return minientropy(a) + minientropy(b) - minientropy(d1) - minientropy(d2)

