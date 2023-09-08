"""
Created by Federico Amato
Calculate the evoltuion of a joint two-cavities system in a phaseonium bath
using the Master Equation formalism.
"""
import cmath
import math
import numpy as np
from qutip import tensor, Qobj, qeye, create, destroy, position, momentum, coherent_dm, thermal_dm, fock_dm
import qutip
import os
import psutil
import tracemalloc


TIMESTEPS = 1000
TIMEDELTA = 1
OMEGA = 0.5  # Strength of Interaction

D = 25

# Ancilla parameters
A = complex(1 / math.sqrt(6), 0)
B = cmath.sqrt(1 - A ** 2)
PHI = np.pi / 2
# Cavities parameters
N1 = 0
N2 = 0


# Class with physic quantities
class Physics:
    def __init__(self, dimension, interaction_time, interaction_strength):
        theta = 1 * interaction_strength * interaction_time
        # Identity
        self.qeye = qeye(dimension)
        # Creation and Annihilation Operators
        self.ad = create(dimension)
        self.a = destroy(dimension)
        self.q = position(dimension)
        self.p = momentum(dimension)

        self.a1 = tensor(self.a, self.qeye)
        self.ad1 = tensor(self.ad, self.qeye)
        self.a2 = tensor(self.qeye, self.a)
        self.ad2 = tensor(self.qeye, self.ad)
        self.q1 = tensor(self.q, self.qeye)
        self.p1 = tensor(self.p, self.qeye)
        self.q2 = tensor(self.qeye, self.q)
        self.p2 = tensor(self.qeye, self.p)
        # Number Operators
        self.ada = self.ad * self.a
        self.aad = self.ad * self.a + 1
        # Ancilla Operators:
        #   Sigma Plus Operator B-
        self.sigmaplus = Qobj([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        #   Sigma Minus Operator B+
        self.sigmaminus = Qobj([
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        # Bosonic Operators
        self.C = (theta * (2 * self.aad).sqrtm()).cosm()
        self.Cp = (theta * (2 * self.ada).sqrtm()).cosm()
        dividend = ((2 * self.aad).sqrtm()).inv()
        sine = (theta * (2 * self.aad).sqrtm()).sinm()
        self.S = self.ad * sine * dividend
        self.Sd = sine * dividend * self.a

        # Interaction
        self.V = tensor(self.a, self.sigmaplus) + tensor(self.ad, self.sigmaminus)
        # Entangled System interactions
        self.V1 = tensor(self.a1, self.sigmaplus) + tensor(self.ad1, self.sigmaminus)
        self.V2 = tensor(self.a2, self.sigmaplus) + tensor(self.ad2, self.sigmaminus)

    @property
    def bosonic_operators(self):
        return [self.C, self.Cp, self.S, self.Sd]

    def kraus_operators_2_cavities(self, ga, gb):
        cc = qutip.tensor(self.C, self.C)
        ssd = qutip.tensor(self.S, self.Sd)
        ek_1 = np.sqrt(ga / 2) * (cc - 2 * ssd)

        scp = qutip.tensor(self.S, self.Cp)
        cs = qutip.tensor(self.C, self.S)
        ek_2 = np.sqrt(ga) * (scp + cs)

        sdc = qutip.tensor(self.Sd, self.C)
        cpsd = qutip.tensor(self.Cp, self.Sd)
        ek_3 = np.sqrt(gb) * (sdc + cpsd)

        cpcp = qutip.tensor(self.Cp, self.Cp)
        sds = qutip.tensor(self.Sd, self.S)
        ek_4 = np.sqrt(gb / 2) * (cpcp - 2 * sds)

        ek_5 = np.sqrt(1 - ga / 2 - gb / 2) * qutip.tensor(self.qeye, self.qeye)

        return [ek_1, ek_2, ek_3, ek_4, ek_5]


p = Physics(dimension=D, interaction_strength=OMEGA, interaction_time=TIMEDELTA)


# Quantum Objects creation
def create_ancilla_qobj(alpha = complex(1/math.sqrt(2), 0),
                        beta = complex(1/math.sqrt(2), 0),
                        phi = np.pi/2,):
    eta = [
        [alpha**2, 0                           , 0                          ],
        [0       , beta**2/2                   , beta**2/2*cmath.exp(1j*phi)],
        [0       , beta**2/2*cmath.exp(-1j*phi), beta**2/2                  ],
    ]
    return Qobj(eta)


def create_system_qobj(dm_type='fock', n_dims=4, **kwargs):
    match dm_type:
        case 'coherent':
            alpha = kwargs.get('alpha', 1)
            state = coherent_dm(n_dims, alpha)
        case 'thermal':
            n = kwargs.get('n', 1)
            state = thermal_dm(n_dims, n)
        case 'fock':
            n = kwargs.get('n', 0)
            state = fock_dm(n_dims, n)
    return state


# UTILITY FUNCTIONS
def _dag(x: np.ndarray):
    return x.conj().T


def dag(x):
    if isinstance(x, np.ndarray):
        return _dag(x)
    elif isinstance(x, list):
        return [_dag(op) for op in x]


def commutator(a: np.ndarray, b: np.ndarray, kind='regular'):
    if kind == 'regular':
        return a @ b - b @ a
    elif kind == 'anti':
        return a @ b + b @ a


def dissipator(x: np.ndarray, system: np.ndarray):
    sandwich = x @ system @ dag(x)
    xdx = dag(x) @ x
    comm = commutator(xdx, system, kind='anti')
    return sandwich - 1/2 * comm


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


def file_dims(filename):
    return int(filename.split('_')[-3][1:])


def file_time(filename):
    return int(filename.split('_')[-2][1:])


def file_timedelta(filename):
    return float(filename.split('_')[-1][2:])


def check_file_metadata(filename):
    dims = file_dims(filename)
    timedelta = file_timedelta(filename)
    return dims == D and timedelta == TIMEDELTA


def create_systems(alpha, beta, phi, n1, n2):
    eta = create_ancilla_qobj(alpha, beta, phi)
    rho1 = create_system_qobj('fock', n=n1, n_dims=D)
    rho2 = create_system_qobj('fock', n=n2, n_dims=D)
    return eta, rho1, rho2


def create_system(rho1, rho2):
    """
    Create a new product state at time 0.
    """
    return np.kron(rho1.full(), rho2.full()).real, 0


def stable_temperature(ga, gb):
    return - 1 / math.log(ga / gb)


def ancilla_parameters(ancilla):
    alpha = cmath.sqrt(ancilla.full()[0, 0])
    beta = cmath.sqrt(2 * ancilla.full()[1, 1])
    phi = math.acos((ancilla.full()[1, 2] / ancilla.full()[1, 1]).real)
    return alpha, beta, phi


def kraus_evolvution(system, kraus_operators):
    return sum(k @ system @ dag(k) for k in kraus_operators)


def meq_evolution(system, ga, gb, operators):
    delta_s = master_equation(system, ga, gb, operators)
    return system + delta_s


def hilbert_is_good(system, check):
    """Check if the Hilbert space truncation is valid"""
    threshold = 9e-4
    if check == 'unitary':
        diagonal_sum = np.sum(system.diagonal())
        return np.abs(diagonal_sum - 1) < threshold
    elif check == 'last_element':
        last_diagonal_element = system.diagonal()[-1]
        return last_diagonal_element < threshold
    else:
        raise ValueError('Check must be either "unitary" or "last_element".')


def main():
    th = OMEGA * TIMEDELTA
    ga = np.real(2 * np.power(A, 2))
    gb = np.real(B ** 2 * (1 + np.cos(PHI)))

    eta, rho1, rho2 = create_systems(A, B, PHI, N1, N2)
    # Create new product state
    rho, t = create_system(rho1, rho2)

    # Create Bosonic operators for ME evolution
    operators = p.bosonic_operators

    rho_evolution = []
    # Profiling
    tracemalloc.start()
    snapshots = []
    memory_usage = []
    disk_usage = []

    # Check if a steady state exists
    if ga / gb < 1:
        print(f'The system will thermalize at temperature {stable_temperature(ga, gb)}.')
    else:
        print('The system will not thermalize.')
    # Evolve
    for t in range(t, t + TIMESTEPS):
        rho = meq_evolution(rho, ga, gb, operators)
        # Take a snapshot of the top 10 memory blocks
        snapshots.append(tracemalloc.take_snapshot().statistics('lineno')[10:])
        # Get the virtual memory and disk usage
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        memory_usage.append(memory.used)
        disk_usage.append(disk.used)
        if not hilbert_is_good(rho, 'unitary'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            rho_evolution.append(rho)

    tracemalloc.stop()
    # Save data
    np.savez(f'rho_evolution_d{D}_t{t + 1}_dt{TIMEDELTA}', *rho_evolution)
    print(f'File saved in {os.getcwd()}\rho_evolution_d{D}_t{t + 1}_dt{TIMEDELTA}.npz')
    # Write profiling file
    with open(f'profiling_d{D}_t{t + 1}_dt{TIMEDELTA}.txt', 'w') as f:
        # Write the contents of the first list to the file
        f.write('Memory:\n')
        for item in memory_usage:
            f.write(f"{item / (1024 * 1024)} MB\n")

        # Write the contents of the second list to the file
        f.write('Disk usage:\n')
        for item in disk_usage:
            f.write(f"{item / (1024 * 1024 * 1024)}GB\n")

        # Write the contents of the third list to the file
        f.write('Memory blocks:\n')
        for tops in snapshots:
            for stat in tops:
                f.write(f"{stat.count} memory blocks allocated at {stat.traceback[0]}\n")
                f.write(f" Total size: {stat.size / 1024} KiB\n")
                f.write(f" Average size: {stat.size / stat.count} bytes\n")


if __name__ == '__main__':
    main()
