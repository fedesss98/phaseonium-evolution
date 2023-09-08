"""
Calculate the evolution of the density matrix of the composite system
with finite interaction time TIMEDELTA.
Can we use the interaction time as control parameter?

TIMEDELTAS
0.01, 0.05, 0.1, 0.5, 1, 5, 10
!!! After 2pi the evolution should repeat itself
"""
import cmath
import math

import utilities as use
from physics import *
from stateobj import Physics
from observables import entropy_vn, purity, covariance

TIMESTEPS = 2000
TIMEDELTA = 1  # Interaction time
OMEGA = 0.1  # Strength of Interaction
D = 17  # Hilbert space dimension

# Ancilla parameters
A = complex(1 / np.sqrt(1 + 2*np.e), 0)
B = cmath.sqrt(1 - A ** 2)
PHI = np.pi / 2
# Cavities parameters
STATE = 'thermal'
N1 = 1
N2 = 1


def file_dims(filename):
    return int(filename.split('_')[-3][1:])


def file_time(filename):
    return int(filename.split('_')[-2][1:])


def file_timedelta(filename):
    return float(filename.split('_')[-1][2:])


def check_file_metadata(filename, d, dt):
    dims = file_dims(filename)
    timedelta = file_timedelta(filename)
    return dims == d and timedelta == dt


def create_systems(alpha, beta, phi, n1, n2, d):
    eta = use.create_ancilla_qobj(alpha, beta, phi)
    rho1 = use.create_system_qobj(STATE, n=n1, alpha=n1, n_dims=d)
    rho2 = use.create_system_qobj(STATE, n=n2, alpha=n2, n_dims=d)
    return eta, rho1, rho2


def load_or_create(rho1, rho2, dims=D, timedelta=TIMEDELTA, create=False):
    """
    Create a new product state or load it from the last saved evolution file.
    Returns the state and the last time step.
    """
    suffix = '.npy'
    return np.kron(rho1.full(), rho2.full()).real, 0


def plot_density_matrix(system, diagonal=False, title=None):
    if diagonal:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(np.diag(system))
    else:
        system = np.real(system)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(system)
    ax.set_title(title)
    plt.show()
    return None


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


def main(dims=20, timedelta=1.0):
    p = Physics(dimension=dims, interaction_strength=OMEGA, interaction_time=timedelta)
    th = OMEGA * timedelta
    print(f'Starting evolution of {dims}-dimensional system with interaction magnitude {th}')

    ga = np.real(2 * A ** 2)
    gb = np.real(B ** 2 * (1 + np.cos(PHI)))

    eta, rho1, rho2 = create_systems(A, B, PHI, N1, N2, dims)
    # Create new product state or load system evolved until time t
    rho, t = load_or_create(rho1, rho2, dims, create=False)
    # plot_density_matrix(rho1, diagonal=True, title='Initial density matrix of the first cavity')
    # plot_density_matrix(rho2, diagonal=True, title='Initial density matrix of the second cavity')
    # plot_density_matrix(rho, diagonal=True, title='Initial density matrix of the composite system')

    # Create Bosonic operators for ME evolution
    operators = p.bosonic_operators
    # Quadrature Operators Vector
    quadratures = [p.q1.full(), p.p1.full(), p.q2.full(), p.p2.full()]

    # entropies = [entropy_vn(rho)]
    # purities = [purity(rho)]
    covariances = [covariance(rho, quadratures)]

    # Check if a steady state exists
    if ga / gb < 1:
        print(f'The system will thermalize at temperature {stable_temperature(ga, gb)}.')
    else:
        print('The system will not thermalize.')

    # Evolve
    timesteps = int(TIMESTEPS / timedelta)
    for t in trange(t, t + timesteps):
        rho = meq_evolution(rho, ga, gb, operators)
        if not hilbert_is_good(rho, 'unitary'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            # entropies.append(entropy_vn(rho))
            # purities.append(purity(rho))
            covariances.append(covariance(rho, quadratures))

    # Save data
    # np.save(f'../objects/{STATE}/rho_entropy_D{dims}_t{t + 1}_dt{TIMEDELTA}', entropies)
    # np.save(f'../objects/{STATE}/rho_purity_D{dims}_t{t + 1}_dt{TIMEDELTA}', purities)
    np.save(f'../objects/{STATE}/rho_covariance_D{dims}_t{t+1}_dt{timedelta}', covariances)

    # Plot final density matrix
    # rho1 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(0).full()
    # plot_density_matrix(rho1, diagonal=True, title='Final density matrix of the first cavity')
    # plot_density_matrix(rho, diagonal=True, title='Final density matrix of the composite system')
    del p


def iter_over_dimensions():
    for dims in [15, 20, 25, 30]:
        main(dims=dims, timedelta=TIMEDELTA)


def iter_over_timedeltas():
    for timedelta in [1.0, 0.5]:
        main(dims=17, timedelta=timedelta)


if __name__ == '__main__':
    iter_over_timedeltas()
