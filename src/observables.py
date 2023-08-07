import numpy as np

try:
    from src.physics import commutator, np_expval, np_anticommutator
except ModuleNotFoundError:
    from physics import commutator, np_expval, np_anticommutator


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


def heat_transfer(delta_rho: np.ndarray, hamiltonian):
    return np.trace(delta_rho @ hamiltonian).real


def cov_matrix_element(rho, op1, op2):
    comm = np_anticommutator(op1, op2)
    return 0.5 * np_expval(rho, comm) - np_expval(rho, op1) * np_expval(rho, op2)


def covariance(rho: np.ndarray, operators):
    cov = [
        [cov_matrix_element(rho, operators[m], operators[n])
         for m in range(len(operators))]
        for n in range(len(operators))
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


def seralian(covariance_matrix: np.ndarray):
    """Serafini, Quantum Continuous Variables; pag. 173; eq. 7.7"""
    i1, i2, i3, i4 = covariance_invariants(covariance_matrix)
    return i1 + i2 + 2 * i3


def seralian_transposed(covariance_matrix: np.ndarray):
    """
    Serafini, Quantum Continuous Variables; pag. 177;
    in Proof of PPT criterion.
    """
    i1, i2, i3, i4 = covariance_invariants(covariance_matrix)
    return i1 + i2 - 2 * i3


def symplectic_eigenvalues(covariance_matrix: np.ndarray):
    """Ferraro, Olivares, Paris - 2005; pag.22"""
    i1, i2, i3, i4 = covariance_invariants(covariance_matrix)
    first_factor = i1 + i2 + 2 * i3
    second_factor = np.sqrt(first_factor ** 2 - 4 * i4)
    d1 = np.sqrt((first_factor + second_factor) / 2)
    d2 = np.sqrt((first_factor - second_factor) / 2)
    return d1, d2


def symplectic_eigenvalues_transposed(covariance_matrix: np.ndarray):
    """
    Serafini, Quantum Continuous Variables; pag. 189; eq. 7.46.
    """
    i1, i2, i3, i4 = covariance_invariants(covariance_matrix)
    first_factor = i1 + i2 - 2 * i3  # Transposition only changes the sign of sigma_AB (i4)
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


def symplectic_temperatures(cov_or_nus: np.ndarray):
    """
    Ferraro, Olivares, Paris - 2005; pag.18
    Every Gaussian state can be obtained from a thermal state by a symplectic transformation.
    This thermal state is a product of thermal states of the single modes,
    with temperature parameters given by symplectic eigenvalues.
    """
    dk = _get_symplectic_eigenvalues(cov_or_nus)
    k2 = 1 / np.sqrt(2)  # Ferraro, Olivares - 2005 pag.1 - pag.14
    f = 1 / (4 * k2**2)
    betak = [np.log((d + 1 + f) / (d - f)) for d in dk]
    return 1 / betak[0], 1 / betak[1]


def symplectic_photons(cov_or_nus: np.ndarray):
    """Ferraro, Olivares, Paris - 2005; pag.19"""
    d1, d2 = _get_symplectic_eigenvalues(cov_or_nus)
    k2 = 1 / np.sqrt(2)  # Ferraro, Olivares - 2005 pag.1 - pag.14
    f = 1 / (4 * k2 ** 2)
    return d1 - f, d2 - f


def mean_photon_numbers(cov: np.ndarray):
    """
    From <n> = q^2 + p^2 - 1/2
    There is a problem with qutip coherent states, which have <n> = q^2 + p^2
    """
    n1 = np.trace(cov[0:2, 0:2])
    n2 = np.trace(cov[2:4, 2:4])
    return n1, n2


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
    return f(np.sqrt(invariants[1])) - f(d1) - f(d2) + f(np.sqrt(e_min))


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


def _partial_transpose(covariance_matrix):
    """
    Serafini, Quantum Continuos Variables, pag.188
    """
    return None


def logarithmic_negativity(covariance_matrix):
    """
    Ferraro, Olivares, Paris - 2005; pag.30
    -------------------------------
    Serafini, Quantum Continuos Variables, pag.187
    """
    k2 = 1 / np.sqrt(2)  # Ferraro, Olivares - 2005 pag.1 - pag.14
    d1, d2 = symplectic_eigenvalues_transposed(covariance_matrix)
    d_minus = min(d1, d2)
    return max(0, -np.log((2 * k2) ** 2 * d_minus))
