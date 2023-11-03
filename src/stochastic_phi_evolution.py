try:
    from rho_evolution import *
except ModuleNotFoundError:
    from src.rho_evolution import *


def _unitary_evolution(system, physic_object, dt):
    rho_d = physic_object.dims
    # Create composite system-ancilla state
    sigma = tensor(
        Qobj(system, dims=[[rho_d, rho_d], [rho_d, rho_d]]),
        physic_object.ancilla
    )
    new_rho = unitary_evolution(sigma, dt, physic_object.V1 + physic_object.V2)
    return new_rho.full()


def _meq_evolution(system, physic_object, interaction_time):
    ga = physic_object.ga
    gb = physic_object.gb
    # Create new bosonic operators with stochastic interaction time
    operators = bosonic_operators(interaction_time, physic_object.a, physic_object.ad)
    return master_equation(system, ga, gb, operators)


def time_evolution(time, dt, experiment, rho, covariances, **kwargs):
    phi_avg = kwargs.get('phi', 2.5881600)
    phis = []
    # Set up random generator with given seed
    rng = np.random.default_rng(seed=int(kwargs.get('seed', 0)))
    for t in time:
        # Extract one phi from a Maxwell distribution
        phi_extracted = use.gaussian_extraction(rng, phi_avg, 0.2)
        phis.append(phi_extracted)
        # Update the experiment object
        experiment._phi = phi_extracted
        # Evolve density matrix
        delta_rho = _meq_evolution(rho, experiment, dt)
        rho = rho + delta_rho
        if not hilbert_is_good(rho, 'last_element'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            covariances.append(covariance(rho, experiment.quadratures))
    return rho, covariances, phis


def save_all_data(dims, timedelta, timesteps, covariances, rho, phis, **kwargs):
    dm_type = kwargs.get('state', 'thermal')
    log_id = kwargs.get('id', '000')
    root_folder = get_root(dm_type)
    save_data(dims, timedelta, timesteps, covariances, rho, **kwargs)
    np.save(
        f'{root_folder}{log_id}_rho_phis_D{dims}_t{timesteps}_dt{timedelta}',
        phis,
    )


def main(dims=25, timedelta=0.8, **kwargs):
    print(f'Starting stochastic evolution of {dims}-dimensional system with interaction time {timedelta}.')
    log_id = kwargs.get('id', '000')
    experiment = setup_experiment(dims, timedelta, **kwargs)
    rho1, rho2 = experiment.systems['rho1'], experiment.systems['rho2']
    # Create new product state and observables or load them evolved until time t
    rho, covariances, heat_transfers, t = load_or_create(experiment, log_id)

    max_timesteps = kwargs.get('max_timesteps', 0)
    if max_timesteps == 0:
        timesteps = int(2000 / timedelta)
        max_timesteps = timesteps
    time = trange(t, t + max_timesteps)
    # Evolve density and save observables
    rho, covariances, phis = time_evolution(
        time, timedelta, experiment, rho, covariances,
        **kwargs
    )
    # Add interaction times to kwargs to save them
    save_all_data(dims, timedelta, t + max_timesteps, covariances, rho, phis, **kwargs)


if __name__ == '__main__':
    d = 25
    dt = 0.8
    alpha = 0.1
    phi = 2.5881600  # Final Temperature = 0.5
    log_id = 'XXXXXX'
    main(d, dt, max_timesteps=20, alpha=alpha, phi=phi, id=log_id)

