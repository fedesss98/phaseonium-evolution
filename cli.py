import click
import numpy as np

from src.rho_evolution import main


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dims', '-d', type=int, default=20, help='Number of dimensions.')
@click.option('--timedelta', '-dt', type=float, default=0.1, help='Time delta.')
@click.argument('kwargs', nargs=-1)
def run_main(dims, timedelta, kwargs):
    """Run the main function with the given dimensions and time delta."""
    kwargs_dict = {}
    for kwarg in kwargs:
        key, value = kwarg.split('=')
        kwargs_dict[key] = value
    main(dims=dims, timedelta=timedelta, **kwargs_dict)


@cli.command()
@click.option('--timedelta', type=float, default=0.1, help='Time delta.')
@click.option('--dims', '-d', multiple=True, default=[20])
@click.argument('kwargs', nargs=-1)
def iter_dims(timedelta, dims, kwargs):
    """Iterate over dimensions with the given time delta."""
    kwargs_dict = {}
    for kwarg in kwargs:
        key, value = kwarg.split('=')
        kwargs_dict[key] = value
    for dim in dims:
        main(dims=dim, timedelta=timedelta, **kwargs_dict)


@cli.command()
@click.option('--dims', '-d', type=int, default=20, help='Number of dimensions.')
@click.option('--alpha', '-a', type=float, default=1/np.sqrt(6), help='Ancilla excited state population')
@click.option('--phi', '-p', type=float, default=np.pi/2, help='Ancilla ground states phase')
@click.option('--state', '-s', type=str, default='thermal', help='Cavities state type')
@click.option('-n1', type=int, default=1, help='Cavity 1 mean photon number')
@click.option('-n2', type=int, default=1, help='Cavity 2 mean photon number')
@click.option('--max-timesteps', type=int, default=0, help='Maximum number of timesteps')
@click.argument('timedeltas', nargs=-1)
def iter_timedeltas(dims, timedeltas, **kwargs):
    """Iterate over time deltas with the given dimensions."""
    for timedelta in timedeltas:
        main(dims=dims, timedelta=float(timedelta), **kwargs)


if __name__ == '__main__':
    cli()
