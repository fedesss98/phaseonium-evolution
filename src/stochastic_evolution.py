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


def meq_evolution(time, dt_mode, experiment, rho, covariances, heat_transfers, exact_evolution):
    for t in time:
        # Extract one interaction time from a Maxwell distribution
        interaction_time = use.maxwell_extraction(dt_mode)
        if exact_evolution:
            rho = _unitary_evolution(rho, experiment, interaction_time)
        else:
            delta_rho = _meq_evolution(rho, experiment, interaction_time)
            rho = rho + delta_rho
        if not hilbert_is_good(rho, 'unitary'):
            print(f'Hilbert space truncation is no more valid at step {t}')
            break
        else:
            covariances.append(covariance(rho, experiment.quadratures))
            # heat_transfers.append(_heat_transfer(delta_rho, experiment))
    return rho, covariances, heat_transfers


def main(dims=20, timedelta=1.0, show_plots=False, **kwargs):
    print(f'Starting stochastic evolution of {dims}-dimensional system with interaction time {timedelta}.')
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
        time, timedelta, experiment, rho, covariances, heat_transfers,
        kwargs.get('exact', False)
    )

    save_data(dims, timedelta, t + max_timesteps, covariances, heat_transfers, rho, **kwargs)

    if show_plots:
        # Trace out evolved cavities
        rho1 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(0).full()
        rho2 = Qobj(rho, dims=[[dims, dims], [dims, dims]]).ptrace(1).full()
        plot_density_matrices(rho1, rho2, rho, t)


if __name__ == '__main__':
    d = 17
    dt = 1.0
    plots = False
    partial = 0
    alpha = complex(1 / np.sqrt(1 + 2*np.e), 0)
    main(d, dt, plots, partial=partial, max_timesteps=500, alpha=alpha)