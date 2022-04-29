import cmath, math
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

from stateobj import QState, QAncilla

from IPython.display import Markdown, display


def print_hello():
    print('Hello world!')


def check_rho_at_time(time, rho_in_time):
    rho = rho_in_time[time]
    partition = 1 / rho.diag()[0].real
    energy_exp = rho.diag()[1].real * partition
    temperature = - 1 / np.log(energy_exp)
    display(Markdown(
        f"### Density Matrix at time ${time}$" if time >= 0 else f"### Last Density Matrix"
    ))
    display(rho)
    print(f'Partition Function = {partition:>}\n'
          f'Coefficients Ratio = {energy_exp:>}\n'
          f'System Temperature = {temperature:>}')    

        
def create_ancilla(alpha = complex(1/math.sqrt(2), 0),
                   beta = complex(1/math.sqrt(2), 0),
                   phi = np.pi/2,):
    return QAncilla(alpha, beta, phi)


def create_system(dm_type='fock', n_dims=4, **kwargs):
    match dm_type:
        case 'coherent':
            alpha = kwargs.get('alpha') if 'alpha' in kwargs else 1
            state = coherent_dm(n_dims, alpha)
        case 'thermal-enr':
            dims = n_dims if isinstance(n_dims, list) else list([n_dims]) 
            excitations = kwargs.get('excitations') if 'excitations' in kwargs else 1
            state = enr_thermal_dm(dims,excitations,n=1)
        case 'thermal':
            n = kwargs.get('n') if 'n' in kwargs else 1
            state = thermal_dm(n_dims, n)
        case 'fock':
            n = kwargs.get('n') if 'n' in kwargs else 0
            state = fock_dm(n_dims, n)
        case 'maxmix':
            state = maximally_mixed_dm(n_dims)
        case 'random':
            seed = kwargs.get('seed') if 'seed' in kwargs else 21
            state = rand_dm(n_dims)
        case 'generic':
            a = kwargs.get('a') if 'a' in kwargs else complex(1, 0)
            b = kwargs.get('b') if 'b' in kwargs else complex(0, 0)
            state = Qobj(np.array([[a, b], [b.conjugate(), 1-a]]))
    return QState(state)

def evolve(state, V, timedelta):
    U = (-1j*V*timedelta).expm()
    total_evolution = U * state * U.dag()
    return total_evolution

def cascade_evolution(eta, joint_system, Vs, timedelta=1e-02):
    """
    Parameters:
    -----------
    systems: array or list of Qobj
    Vs: array or list
        Interaction system-ancilla
    """
    total_system = tensor(joint_system, eta)
    Us = []
    for V in Vs:
        U = (-1j*V*timedelta).expm()
        total_system = U * total_system * U.dag()
    return total_system


def interact(system, ancilla, interactions, time):
        """
        Unitary Evolution.
        Parameters
        ----------
        ancilla: QAncilla (Qobj)
        interactions: array or list
        time: float
        """
        unitaries = [(-1j*interaction*time).expm() for interaction in interactions]
        total_system = tensor(system, ancilla)
        for U in unitaries:
            total_system = U * total_system * U.dag()
        return total_system

def get_temperature(rho, energy):
    """Find kT from Boltzmann distribution"""
    return - energy / np.log(rho.diag()[1].real / rho.diag()[0].real)


def get_temperature_from_eta(alpha, beta, phi, energy):
    """Find the final stable Temperature of the System from the Ancilla State
    from the ratio Gamma\betha / Gamma\alpha"""
    gamma_ratio = 2*abs(alpha)**2 / (abs(beta)**2*(1 + math.cos(phi)))
    return energy / math.log(gamma_ratio)

    
def is_thermal(rho):
    """The Density Matrix is thermal if it is diagonal and 
    the diagonal elements are sorted from bigger to smaller"""
    in_diag = rho.diag()
    out_of_diag = rho - np.diag(in_diag)
    if not np.count_nonzero(out_of_diag) and np.all(np.diff(in_diag) <= 0):
        return True
    else:
        return False
    
    
if __name__ == "__main__":
    print_hello()