import click
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
@click.option('--dims', type=int, default=20, help='Number of dimensions.')
@click.option('--timedeltas', '-dt', multiple=True, default=[1.0], help='Time deltas.')
@click.argument('kwargs', nargs=-1)
def iter_timedeltas(dims, timedeltas, kwargs):
    """Iterate over time deltas with the given dimensions."""
    kwargs_dict = {}
    if len(kwargs) > 0:
        click.echo(kwargs)
        for kwarg in kwargs:
            key, value = kwarg.split('=')
            kwargs_dict[key] = value
    for timedelta in timedeltas:
        main(dims=dims, timedelta=timedelta, **kwargs_dict)


if __name__ == '__main__':
    cli()
