"""
Calculate the evolution of the density matrix of the composite system
with finite interaction time TIMEDELTA.
Can we use the interaction time as control parameter?

TIMEDELTAS
0.01, 0.05, 0.1, 0.5, 1, 5, 10
!!! After 2pi the evolution should repeat itself
For the paper we use:
0.05 - 20'000 iterations
0.1 - 6'000 iterations
1.0 - 1000 iterations

In simulations, with phi=pi/2 we have:
    KT = 1 -> alpha = 1/Sqrt[1 + 2 e]
    KT = 2 -> alpha = 1/Sqrt[1 + 2 e^(1/2)]
"""
import cmath
import math

import numpy as np

try:
    import utilities as use
    from physics import *
    from stateobj import Physics
    from observables import entropy_vn, purity, covariance, heat_transfer
    from utilities import default_alpha, default_phi
except ModuleNotFoundError:
    import src.utilities as use
    from src.physics import *
    from src.stateobj import Physics
    from src.observables import entropy_vn, purity, covariance, heat_transfer
    from src.utilities import default_alpha, default_phi


def setup_experiment(dims, timedelta, **kwargs):
    omega = kwargs.get('omega', 0.5)
    # Ancilla parameters
    alpha = kwargs.get('alpha') if kwargs.get('alpha') is not None else default_alpha()
    phi = kwargs.get('phi') if kwargs.get('phi') is not None else default_phi()
    print(f'alpha = {alpha}, phi = {phi}')
    # Cavities parameters
    state = kwargs.get('state', 'thermal')
    n1 = kwargs.get('n1', 1)
    n2 = kwargs.get('n2', 1)
    experiment = Physics(dimension=dims,
                         interaction_strength=omega, interaction_time=timedelta,
                         alpha=alpha,
                         phi=phi)
    experiment.create_system(state, n=n1, alpha=n1, name='rho1')
    experiment.create_system(state, n=n2, alpha=n1, name='rho2')
    return experiment


def get_root(dm_type):
    try:
        root = f'../objects/{dm_type}/'
        os.listdir(root)
    except FileNotFoundError:
        root = f'objects/{dm_type}/'
    return root


def file_dims(filename):
    return int(filename.split('_')[-3][1:])


def file_time(filename):
    return int(filename.split('_')[-2][1:])


def file_timedelta(filename):
    return float(filename.split('_')[-1][2:])


def check_file(filename, d, dt):
    filename = filename.removesuffix('.npy')
    try:
        dims = file_dims(filename)
        timedelta = file_timedelta(filename)
    except IndexError:
        return False
    else:
        return filename.startswith('rho_last_') and dims == d and timedelta == dt


def load_or_create(experiment, create=False):
    """
    Create a new product state or load it from the last saved evolution file.
    Returns the state and the last time step.
    """
    suffix = '.npy'
    rho1 = experiment.systems['rho1']['density']
    rho2 = experiment.systems['rho2']['density']
    dm_type = experiment.systems['rho1']['type']
    dims = experiment.dims
    dt = experiment.dt
    root_folder = get_root(dm_type)
    files = [f.removesuffix(suffix) for f in os.listdir(root_folder) if check_file(f, dims, dt)]
    times = [file_time(f) for f in files]
    if files and not create:
        # There are files to load
        last_t = max(times)
        rho = np.load(root_folder + f'rho_last_D{dims}_t{last_t}_dt{dt}' + suffix)
        covariances = np.load(root_folder + f'rho_covariance_D{dims}_t{last_t}_dt{dt}' + suffix).tolist()
        heats = np.load(root_folder + f'rho_heats_D{dims}_t{last_t}_dt{dt}' + suffix).tolist()
        print(f'Saved files exists until time {last_t}')
        return rho, covariances, heats, last_t
    else:
        rho = np.kron(rho1.full(), rho2.full()).real
        covariances = [covariance(rho, experiment.quadratures)]
        heats = [(0, 0, 0)]
        return rho, covariances, heats, 0


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
    return master_equation(system, ga, gb, operators)


def _heat_transfer(dr, experiment):
    h1 = experiment.h1.full()
    h2 = experiment.h2.full()
    h = h1 + h2
    j1 = heat_transfer(dr, h1)
    j2 = heat_transfer(dr, h2)
    jc = j1 + j2
    return [j1, j2, jc]


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
    rho1, rho2 = experiment.systems['rho1'], experiment.systems['rho2']
    # Create new product state and observables or load them evolved until time t
    rho, covariances, heat_transfers, t = load_or_create(experiment)
    if show_plots:
        plot_density_matrix(rho1['density'], diagonal=True, title='Initial density matrix of the first cavity')
        plot_density_matrix(rho2['density'], diagonal=True, title='Initial density matrix of the second cavity')
        plot_density_matrix(rho, diagonal=True, title='Initial density matrix of the composite system')

    # Check if a steady state exists
    if experiment.ga / experiment.gb < 1:
        print(f'The system will thermalize at temperature {experiment.stable_temperature}.')
    else:
        print('The system will not thermalize.')

    # Evolve
    total_time_range = 2000  # Approximate time to thermalize the cavities
    timesteps = int(total_time_range / timedelta)
    max_timesteps = kwargs.get('max_timesteps', 0)
    if kwargs.get('max_timesteps', 0) == 0:
        max_timesteps = timesteps
    for t in trange(t, t + max_timesteps):
        delta_rho = meq_evolution(rho, experiment)
        rho = rho + delta_rho
        if not hilbert_is_good(rho, 'unitary'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            covariances.append(covariance(rho, experiment.quadratures))
            heat_transfers.append(_heat_transfer(delta_rho, experiment))

    # Save data
    root_folder = get_root(kwargs.get('state', 'thermal'))
    np.save(root_folder + f'rho_covariance_D{dims}_t{t+1}_dt{timedelta}', covariances)
    np.save(root_folder + f'rho_heats_D{dims}_t{t+1}_dt{timedelta}', heat_transfers)
    np.save(root_folder + f'rho_last_D{dims}_t{t+1}_dt{timedelta}', rho)

    if show_plots:
        # Plot final density matrix
        # Trace out the second cavity
        rho1 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(0).full()
        plot_density_matrix(rho1, diagonal=True, title='Final density matrix of the first cavity')
        plot_density_matrix(rho, diagonal=True, title='Final density matrix of the composite system')


if __name__ == '__main__':
    main()
