import click

from src.rho_evolution import main as rho_evolution
from src.stochastic_evolution import main as stochastic_evolution


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dims', '-d', type=int, default=20, help='Number of dimensions.')
@click.option('--timedelta', '-dt', type=float, default=0.1, help='Interaction time.')
@click.option('--omega', type=float, default=1.0, help='Interaction strength.')
@click.option('--alpha', '-a', type=float, help='Ancilla excited state population')
@click.option('--phi', '-p', type=float, help='Ancilla ground states phase')
@click.option('--state', '-s', type=str, default='thermal', help='Cavities state type')
@click.option('-n1', type=float, default=1, help='Cavity 1 mean photon number')
@click.option('-n2', type=float, default=1, help='Cavity 2 mean photon number')
@click.option('--max-timesteps', type=int, default=0, help='Maximum number of timesteps')
@click.option('--partial', type=int, default=0, help='Save partial evolution')
@click.option('-id', type=str, default='000', help='Log ID for the simulation')
@click.option('--exact/--not-exact', default=False, help='Log ID for the simulation')
def simulate(dims, timedelta, **kwargs):
    """Run the main function with the given dimensions and time delta."""
    rho_evolution(dims=dims, timedelta=timedelta, **kwargs)


@cli.command()
@click.option('--dims', '-d', type=int, default=20, help='Number of dimensions.')
@click.option('--timedelta', '-dt', type=float, nargs=2, default=(1.0, 0.1), help='Average Interaction Time or (Avg, StDev).')
@click.option('--omega', type=float, default=1.0, help='Interaction strength.')
@click.option('--alpha', '-a', type=float, help='Ancilla excited state population')
@click.option('--phi', '-p', type=float, help='Ancilla ground states phase')
@click.option('--state', '-s', type=str, default='thermal', help='Cavities state type')
@click.option('-n1', type=float, default=1, help='Cavity 1 mean photon number')
@click.option('-n2', type=float, default=1, help='Cavity 2 mean photon number')
@click.option('--max-timesteps', type=int, default=0, help='Maximum number of timesteps')
@click.option('--partial', type=int, default=0, help='Save partial evolution')
@click.option('-id', type=str, default='000', help='Log ID for the simulation')
@click.option('-seed', type=str, default='0001', help='Log ID for the simulation')
@click.option('--exact/--not-exact', default=False, help='Log ID for the simulation')
@click.argument('distribution', default='gaussian')
def stochastic_simulate(dims, timedelta, **kwargs):
    """Run the main function with the given dimensions and stochastic time deltas."""
    stochastic_evolution(dims=dims, timedelta=timedelta, **kwargs)


@cli.command()
@click.option('--dims', '-d', type=int, default=25, help='Number of dimensions.')
@click.option('--timedelta', '-dt', type=float, default=0.8, help='Interaction time.')
@click.option('--omega', type=float, default=1.0, help='Interaction strength.')
@click.option('--alpha', '-a', default=0.1, type=float, help='Ancilla excited state population')
@click.option('--phi', '-p', type=float, help='Ancilla ground states phase')
@click.option('--state', '-s', type=str, default='thermal', help='Cavities state type')
@click.option('-n1', type=float, default=0.5819767, help='Cavity 1 mean photon number')
@click.option('-n2', type=float, default=0.5819767, help='Cavity 2 mean photon number')
@click.option('--max-timesteps', type=int, default=300, help='Maximum number of timesteps')
@click.option('-id', type=str, default='000', help='Log ID for the simulation')
@click.option('-seed', type=str, default='0001', help='Log ID for the simulation')
def stochastic_phase_simulate(dims, timedelta, **kwargs):
    """Run the main function with the given dimensions and stochastic time deltas."""
    stochastic_evolution(dims=dims, timedelta=timedelta, **kwargs)


@cli.command()
@click.option('--timedelta', type=float, default=0.1, help='Time delta.')
@click.option('--dims', '-d', multiple=True, default=[20])
@click.argument('kwargs', nargs=-1)
def iter_dims(timedelta, dims, **kwargs):
    """Iterate over dimensions with the given time delta."""
    for dim in dims:
        rho_evolution(dims=dim, timedelta=timedelta, **kwargs)


@cli.command()
@click.option('--dims', '-d', type=int, default=20, help='Number of dimensions.')
@click.option('--omega', type=float, default=1.0, help='Interaction strength.')
@click.option('--alpha', '-a', type=float, help='Ancilla excited state population')
@click.option('--phi', '-p', type=float, help='Ancilla ground states phase')
@click.option('--state', '-s', type=str, default='thermal', help='Cavities state type')
@click.option('-n1', type=float, default=1, help='Cavity 1 mean photon number')
@click.option('-n2', type=float, default=1, help='Cavity 2 mean photon number')
@click.option('--max-timesteps', type=int, default=0, help='Maximum number of timesteps')
@click.option('--partial', type=int, default=0, help='Save partial evolution')
@click.argument('timedeltas', nargs=-1)
def iter_timedeltas(dims, timedeltas, **kwargs):
    """Iterate over time deltas with the given dimensions."""
    for timedelta in timedeltas:
        rho_evolution(dims=dims, timedelta=float(timedelta), **kwargs)


if __name__ == '__main__':
    cli()
