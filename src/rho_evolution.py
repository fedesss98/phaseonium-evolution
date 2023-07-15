"""
Calculate the evolution of the density matrix of the composite system
with finite interaction time TIMEDELTA.
Can we use the interaction time as control parameter?

TIMEDELTAS
1e-01
2e-01
3e-01
1
2
1e-02
1e-03
"""
import cmath
import math

import utilities as use
from physics import *
from stateobj import Physics

TIMESTEPS = 2000
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

p = Physics(dimension=D, interaction_strength=OMEGA, interaction_time=TIMEDELTA)


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
    eta = use.create_ancilla_qobj(alpha, beta, phi)
    rho1 = use.create_system_qobj('fock', n=n1, n_dims=D)
    rho2 = use.create_system_qobj('fock', n=n2, n_dims=D)
    return eta, rho1, rho2


def load_or_create(rho1, rho2, create=False):
    """
    Create a new product state or load it from the last saved evolution file.
    Returns the state and the last time step.
    """
    if create:
        return np.kron(rho1.full(), rho2.full()).real, 0
    try:
        files = [file.removesuffix('.npz') for file in os.listdir('../objects') if file.endswith('.npz')]
        times = [file_time(file) for file in files if check_file_metadata(file)]
        t = max(times) if len(times) > 0 else 0
        zipped_evolution = np.load(f'../objects/rho_evolution_d{D}_t{t}_dt{TIMEDELTA}.npz')
        rho = zipped_evolution[zipped_evolution.files[-1]]
        del zipped_evolution
        print(f'Loaded evolution until step {t}.')
    except FileNotFoundError:
        rho = np.kron(rho1.full(), rho2.full()).real
        t = 0
        print('File not found. Starting a new evolution.')
    return rho, t


def plot_density_matrix(system, diagonal=False, title=None):
    if diagonal:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(np.diag(system))
        ax.set_title(title)
        plt.show()
    else:
        system = np.real(system)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(system)
        ax.set_title(title)
        plt.show()
    return None


def stable_temperature(ga, gb):
    temperature = - 1 / math.log(ga / gb)
    return temperature


def ancilla_parameters(ancilla):
    alpha = cmath.sqrt(ancilla.full()[0, 0])
    beta = cmath.sqrt(2 * ancilla.full()[1, 1])
    phi = math.acos((ancilla.full()[1, 2] / ancilla.full()[1, 1]).real)
    return alpha, beta, phi


def kraus_evolvution(system, kraus_operators):
    new_system = 0
    for k in kraus_operators:
        new_system += k @ system @ dag(k)
    return new_system


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
    print(f'Starting evolution of {D}-dimensional system')
    th = OMEGA * TIMEDELTA
    ga = np.real(2 * A ** 2)
    gb = np.real(B ** 2 * (1 + np.cos(PHI)))

    eta, rho1, rho2 = create_systems(A, B, PHI, N1, N2)
    # Create new product state or load system evolved until time t
    rho, t = load_or_create(rho1, rho2, create=False)
    plot_density_matrix(rho1, diagonal=True, title='Initial density matrix of the first cavity')
    plot_density_matrix(rho2, diagonal=True, title='Initial density matrix of the second cavity')
    plot_density_matrix(rho, diagonal=True, title='Initial density matrix of the composite system')

    # Create Kraus operators
    kraus_operators = [k.full() for k in p.kraus_operators_2_cavities(ga, gb)]
    # Create Bosonic operators for ME evolution
    operators = p.bosonic_operators

    rho_evolution = list()

    # Check if a steady state exists
    if ga / gb < 1:
        print(f'The system will thermalize at temperature {stable_temperature(ga, gb)}.')
    else:
        print('The system will not thermalize.')
    # Evolve
    for t in trange(t, t + TIMESTEPS):
        rho = meq_evolution(rho, ga, gb, operators)
        if not hilbert_is_good(rho, 'unitary'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            rho_evolution.append(rho)

    # Save data
    np.savez(f'../objects/rho_evolution_d{D}_t{t + 1}_dt{TIMEDELTA}', *rho_evolution)
    # Plot final density matrix
    rho1 = Qobj(rho, dims=[[D, D], [D, D]]).ptrace(0).full()
    plot_density_matrix(rho1, diagonal=True, title='Final density matrix of the first cavity')
    plot_density_matrix(rho, diagonal=True, title='Final density matrix of the composite system')


if __name__ == '__main__':
    main()
