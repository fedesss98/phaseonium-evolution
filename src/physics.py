import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj
import os
from tqdm import tqdm, trange


def _dag(x: Qobj | np.ndarray):
    if isinstance(x, Qobj):
        return x.dag()
    elif isinstance(x, np.ndarray):
        return x.conj().T


def dag(x):
    if isinstance(x, (Qobj, np.ndarray)):
        return _dag(x)
    elif isinstance(x, list):
        return [_dag(op) for op in x]


def commutator(a: Qobj | np.ndarray, b: Qobj | np.ndarray, kind='regular'):
    if isinstance(a, Qobj) and isinstance(b, Qobj):
        if kind == 'regular':
            return a * b - b * a
        elif kind == 'anti':
            return a * b + b * a
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if kind == 'regular':
            return a @ b - b @ a
        elif kind == 'anti':
            return a @ b + b @ a
    else:
        raise TypeError('a and B must be of the same type')


def np_commutator(a: np.ndarray, b: np.ndarray):
    return a @ b - b @ a


def np_anticommutator(a: np.ndarray, b: np.ndarray):
    return a @ b + b @ a


def np_expval(rho: np.ndarray, obs: np.ndarray):
    return np.trace(rho @ obs)


def dissipator(x: Qobj | np.ndarray, system: Qobj | np.ndarray, kind='regular'):
    if kind == 'regular':
        if isinstance(system, Qobj):
            sandwich = x * system * dag(x)
        else:
            sandwich = x @ system @ dag(x)
    elif kind == 'anti':
        if isinstance(system, Qobj):
            sandwich = dag(x) * system * x
        else:
            sandwich = dag(x) @ system @ x
    else:
        raise ValueError('Dissipator kind must be "regular" or "anti"')
    xdx = dag(x) @ x if isinstance(x, np.ndarray) else dag(x) * x
    comm = commutator(xdx, system, kind='anti')
    return sandwich - 1/2 * comm


def bosonic_operators(theta, a, ad):
    # Number Operators
    ada = ad * a
    aad = ad * a + 1
    c = (theta * (2 * aad).sqrtm()).cosm()
    cp = (theta * (2 * ada).sqrtm()).cosm()
    dividend = ((2 * aad).sqrtm()).inv()
    sine = (theta * (2 * aad).sqrtm()).sinm()
    s = ad * sine * dividend
    sd = sine * dividend * a
    return c, cp, s, sd


def master_equation(system, ga, gb, operators):
    c, cp, s, sd = operators

    # Dissipators
    d_cc_2ssd = dissipator(np.kron(c, c) - 2*np.kron(s, sd), system)
    d_cs_scp = dissipator(np.kron(c, s) + np.kron(s, cp), system)
    first_line = 0.5 * d_cc_2ssd + d_cs_scp

    d_cpcp_2sds = dissipator(np.kron(cp, cp) - 2*np.kron(sd, s), system)
    d_cp_sd = dissipator(np.kron(cp, sd) + np.kron(sd, c), system)
    second_line = 0.5 * d_cpcp_2sds + d_cp_sd
    return ga * first_line + gb * second_line


def _partial_transpose(covariance_matrix, subsystem=0):
    """
    Serafini, Quantum Continuos Variables, pag.188
    """
    T = np.diag([(-1) ** i for i in range(covariance_matrix.shape[0])])
    return T


def unitary_evolution(system: Qobj, interaction_time, interaction_H: Qobj):
    evolution_exponent = -1j * interaction_time * interaction_H
    evolution_operator = evolution_exponent.expm()

    # Evolve composite state
    new_system = evolution_operator * system * evolution_operator.dag()
    new_rho = new_system.ptrace([0, 1])
    return new_rho

