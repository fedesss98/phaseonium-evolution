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

try:
    import utilities as use
    from physics import *
    from stateobj import Physics
    from observables import entropy_vn, purity, covariance
except ModuleNotFoundError:
    import src.utilities as use
    from src.physics import *
    from src.stateobj import Physics
    from src.observables import entropy_vn, purity, covariance


def setup_experiment(dims, timedelta, **kwargs):
    omega = 0.5 if 'omega' not in kwargs else kwargs['omega']
    # Ancilla parameters
    alpha = complex(1 / math.sqrt(6), 0) if 'alpha' not in kwargs else kwargs['alpha']
    beta = cmath.sqrt(1 - alpha ** 2) if 'beta' not in kwargs else kwargs['beta']
    phi = np.pi / 2 if 'phi' not in kwargs else kwargs['phi']
    # Cavities parameters
    state = 'thermal' if 'state' not in kwargs else kwargs['state']
    n1 = 1 if 'n1' not in kwargs else kwargs['n1']
    n2 = 1 if 'n2' not in kwargs else kwargs['n2']
    experiment = Physics(dimension=dims,
                         interaction_strength=omega, interaction_time=timedelta,
                         alpha=alpha,
                         beta=beta,
                         phi=phi)
    experiment.create_system(state, n=n1, alpha=n1, name='rho1')
    experiment.create_system(state, n=n2, alpha=n1, name='rho2')
    return experiment


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


def load_or_create(rho1, rho2):
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
        ax.set_title(title)
        plt.show()
    else:
        system = np.real(system)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(system)
        ax.set_title(title)
        plt.show()
    return None


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


def meq_evolution(system, physic_object):
    ga = physic_object.ga
    gb = physic_object.gb
    operators = physic_object.bosonic_operators
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


def main(dims=20, timedelta=1.0, show_plots=False, **kwargs):
    print(f'Starting evolution of {dims}-dimensional system with interaction time {timedelta}.')

    experiment = setup_experiment(dims, timedelta, **kwargs)
    rho1 = experiment.systems['rho1']
    rho2 = experiment.systems['rho2']
    # Create new product state or load system evolved until time t
    rho, t = load_or_create(rho1['density'], rho2['density'])
    if show_plots:
        plot_density_matrix(rho1['density'], diagonal=True, title='Initial density matrix of the first cavity')
        plot_density_matrix(rho2['density'], diagonal=True, title='Initial density matrix of the second cavity')
        plot_density_matrix(rho, diagonal=True, title='Initial density matrix of the composite system')

    # Check if a steady state exists
    if experiment.ga / experiment.gb < 1:
        print(f'The system will thermalize at temperature {experiment.stable_temperature}.')
    else:
        print('The system will not thermalize.')

    # Quadrature Operators Vector
    quadratures = [experiment.q1.full(), experiment.p1.full(),
                   experiment.q2.full(), experiment.p2.full()]

    # Create covariance evolution vector
    covariances = [covariance(rho, quadratures)]

    # Evolve
    total_time_range = 2000  # Approximate time to thermalize the cavities
    timesteps = int(total_time_range / timedelta)
    for t in trange(t, t + timesteps):
        rho = meq_evolution(rho, experiment)
        if not hilbert_is_good(rho, 'unitary'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            covariances.append(covariance(rho, quadratures))

    # Save data
    np.save(f'../objects/{rho1["type"]}/rho_covariance_D{dims}_t{t+1}_dt{timedelta}', covariances)

    if show_plots:
        # Plot final density matrix
        # Trace out the second cavity
        rho1 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(0).full()
        plot_density_matrix(rho1, diagonal=True, title='Final density matrix of the first cavity')
        plot_density_matrix(rho, diagonal=True, title='Final density matrix of the composite system')


if __name__ == '__main__':
    main()
