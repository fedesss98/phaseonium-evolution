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

Try to change alpha and phi as to have the same KT but with different dynamics
(Kraus operators does change with gamma alpha and gamma beta).

KT = 1
____________________
    phi   |   alpha
       0  |  0.518596    1/Sqrt[1 + E]
    pi/6  |  0.505499    1/Sqrt[1 - 4 (-2 + Sqrt[3]) E]
    pi/4  |  0.488843    1/Sqrt[1 - 2 (-2 + Sqrt[2]) E]
    pi/3  |  0.465022    Sqrt[3/(3 + 4 E)]
    pi/2  |  0.394160    1/Sqrt[1 + 2 E]
      pi  |              inf

KT = 2
____________________
    phi   |   alpha
       0  |  0.614443    1/Sqrt[1 + Sqrt[E]]
    pi/6  |  0.601157    1/Sqrt[1 - 4 (-2 + Sqrt[3]) Sqrt[E]]
    pi/4  |  0.584047    1/Sqrt[1 - 2 (-2 + Sqrt[2]) Sqrt[E]]
    pi/3  |  0.559166    Sqrt[3/(3 + 4 Sqrt[E])]
    pi/2  |  0.482386    1/Sqrt[1 + 2 Sqrt[E]]
      pi  |              inf

Partial evolution should highlight real-evolution stroboscopic points.
"""
import cmath
import math

import pandas as pd
from qutip import Qobj, tensor

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
    omega = kwargs.get('omega', 1)
    # Ancilla parameters
    alpha = kwargs.get('alpha', default_alpha())
    phi = kwargs.get('phi', default_phi())
    # Cavities parameters
    state = kwargs.get('state', 'thermal')
    n1 = kwargs.get('n1', 1)
    n2 = kwargs.get('n2', 1)
    experiment = Physics(dimension=dims,
                         interaction_strength=omega, interaction_time=timedelta,
                         alpha=alpha,
                         phi=phi)
    experiment.create_system(state, n=n1, alpha=n1, name='rho1')
    experiment.create_system(state, n=n2, alpha=n2, name='rho2')

    # Check if a steady state exists
    if experiment.ga / experiment.gb < 1:
        print(f'The system will thermalize at temperature {experiment.stable_temperature}.')
    else:
        print('The system will not thermalize.')

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


def check_file(filename, d, dt, log_id='000'):
    filename = filename.removesuffix('.npy')
    try:
        dims = file_dims(filename)
        timedelta = file_timedelta(filename)
    except IndexError:
        return False
    else:
        if log_id != '000':
            return filename.startswith(f'{log_id}_rho_last_') and dims == d and timedelta == dt
        else:
            return filename.startswith('rho_last_') and dims == d and timedelta == dt


def load_or_create(experiment, log_id, create=False):
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
    files = [f.removesuffix(suffix) for f in os.listdir(root_folder) if check_file(f, dims, dt, log_id)]
    times = [file_time(f) for f in files]
    if files and not create:
        # There are files to load
        last_t = max(times)
        log_id = log_id + '_' if log_id != '000' else ''
        rho = np.load(root_folder + log_id + f'rho_last_D{dims}_t{last_t}_dt{dt}' + suffix)
        covariances = np.load(root_folder + log_id + f'rho_covariance_D{dims}_t{last_t}_dt{dt}' + suffix).tolist()
        try:
            heats = np.load(root_folder + log_id + f'rho_heats_D{dims}_t{last_t}_dt{dt}' + suffix).tolist()
        except FileNotFoundError:
            print("Heats not loaded")
            heats = None
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
    else:
        system = np.real(system)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(system)
    ax.set_title(title)
    plt.show()
    return None


def plot_density_matrices(rho1, rho2, rho, t):
    plot_density_matrix(rho1, diagonal=True, title=f'Density matrix of the first cavity at time {t}')
    plot_density_matrix(rho2, diagonal=True, title=f'Density matrix of the second cavity at time {t}')
    plot_density_matrix(rho, diagonal=True, title=f'Density matrix of the composite system at time {t}')
    return None


def ancilla_parameters(ancilla):
    alpha = cmath.sqrt(ancilla.full()[0, 0])
    beta = cmath.sqrt(2 * ancilla.full()[1, 1])
    phi = math.acos((ancilla.full()[1, 2] / ancilla.full()[1, 1]).real)
    return alpha, beta, phi


def _kraus_evolvution(system, physic_object):
    kraus_operators = physic_object.kraus_operators_2_cavities()
    dims = physic_object.dims
    new_system = Qobj()
    system = Qobj(system, dims=[[dims, dims], [dims, dims]])
    for k in kraus_operators:
        new_system += k * system * dag(k)
    return new_system.full()


def _unitary_evolution(system, physic_object: Physics):
    rho_d = physic_object.dims
    # Create composite system-ancilla state
    sigma = tensor(
        Qobj(system, dims=[[rho_d, rho_d], [rho_d, rho_d]]),
        physic_object.ancilla
    )
    dt = physic_object.dt
    new_rho = unitary_evolution(sigma, dt, physic_object.V1 + physic_object.V2)
    return new_rho.full()


def _meq_evolution(system, physic_object):
    ga = physic_object.ga
    gb = physic_object.gb
    operators = physic_object.bosonic_operators
    return master_equation(system, ga, gb, operators)


def _partial_evolution(system, physic_object, steps_per_timestep=3):
    partial_covariances = []
    dt = physic_object.theta / steps_per_timestep
    evolution_exponent = -1j * (dt * physic_object.V1 + dt * physic_object.V2)
    evolution_operator = evolution_exponent.expm()
    rho_d = physic_object.dims
    sigma = tensor(
        Qobj(system, dims=[[rho_d, rho_d], [rho_d, rho_d]]),
        physic_object.ancilla
    )
    for _ in range(steps_per_timestep - 1):
        sigma = evolution_operator * sigma * evolution_operator.dag()
        partial_rho = sigma.ptrace([0, 1])
        partial_covariances.append(covariance(partial_rho.full(), physic_object.quadratures))
    return partial_covariances


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
    threshold = 1e-9
    if check == 'unitary':
        diagonal_sum = np.sum(system.diagonal())
        return np.abs(diagonal_sum - 1) < threshold
    elif check == 'last_element':
        last_diagonal_element = system.diagonal()[-1]
        return last_diagonal_element < threshold
    else:
        raise ValueError('Check must be either "unitary" or "last_element".')


def meq_evolution(time, experiment, rho, covariances, heat_transfers, partial, exact_evolution=False):
    for t in time:
        if partial:
            partial_covariances = _partial_evolution(rho, experiment, partial)
            covariances.extend(partial_covariances)
        if exact_evolution:
            old_rho = rho
            rho = _kraus_evolvution(old_rho, experiment)
            delta_rho = rho - old_rho
        else:
            delta_rho = _meq_evolution(rho, experiment)
            rho = rho + delta_rho
        if not hilbert_is_good(rho, 'last_element'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            covariances.append(covariance(rho, experiment.quadratures))
            heat_transfers.append(_heat_transfer(delta_rho, experiment))
    return rho, covariances, heat_transfers


def save_data(dims, timedelta, t, covariances, last_rho, heat_transfers=None, **kwargs):
    dm_type = kwargs.get('state', 'thermal')
    root_folder = get_root(dm_type)
    log_id = kwargs.get('id', '000')
    if log_id == '000':
        log_id = use.get_last_id(root_folder)
    # Save data
    np.save(root_folder + f'{log_id}_rho_covariance_D{dims}_t{t}_dt{timedelta}', covariances)
    if heat_transfers is not None:
        np.save(root_folder + f'{log_id}_rho_heats_D{dims}_t{t}_dt{timedelta}', heat_transfers)
    np.save(root_folder + f'{log_id}_rho_last_D{dims}_t{t}_dt{timedelta}', last_rho)


def main(dims=20, timedelta=1.0, show_plots=False, **kwargs):
    print(f'Starting evolution of {dims}-dimensional system with interaction time {timedelta}.')
    log_id = kwargs.get('id', '000')
    experiment = setup_experiment(dims, timedelta, **kwargs)
    rho1, rho2 = experiment.systems['rho1'], experiment.systems['rho2']
    # Create new product state and observables or load them evolved until time t
    rho, covariances, heat_transfers, t = load_or_create(experiment, log_id)
    if show_plots:
        plot_density_matrices(rho1, rho2, rho, t)

    max_timesteps = kwargs.get('max_timesteps', 0)
    if kwargs.get('max_timesteps', 0) == 0:
        timesteps = int(2000 / timedelta)
        max_timesteps = timesteps
    time = trange(t, t + max_timesteps)
    # Evolve density and save observables
    rho, covariances, heat_transfers = meq_evolution(
        time, experiment, rho, covariances, heat_transfers,
        kwargs.get('partial', 0), kwargs.get('exact', False)
    )

    save_data(dims, timedelta, t + max_timesteps, covariances, rho, heat_transfers, **kwargs)

    if show_plots:
        # Trace out evolved cavities
        rho1 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(0).full()
        rho2 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(1).full()
        plot_density_matrices(rho1, rho2, rho, t)


if __name__ == '__main__':
    d = 25
    dt = 0.4
    kwargs = {
        'plots': False,
        'partial': 0,
        'alpha': complex(1 / np.sqrt(1 + 2 * np.e), 0),
        'exact': False,
        'state': 'coherent',
        'n1': 1,
        'n2': 0,
    }

    main(d, dt, **kwargs)
