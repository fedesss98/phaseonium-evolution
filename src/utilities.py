import cmath, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qutip import *

try:
    from src.stateobj import QState, QAncilla
    from src.stateobj import Physics as p
except ModuleNotFoundError:
    from stateobj import QState, QAncilla
    from stateobj import Physics as p

from IPython.display import Markdown, display


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def _repr_latex_(self):
        return np_display(self.matrix)


def default_alpha():
    return complex(1 / math.sqrt(1 + 2*np.e), 0)


def default_phi():
    return np.pi / 2


def np_display(a: np.ndarray):
    if a.ndim == 1:
        raise ValueError("Cannot display 1D array")

    M, N = a.shape

    s = r'\begin{equation*}\left(\begin{array}{*{11}c}'

    def _format_float(value):
        if value == 0.0:
            return "0.0"
        elif abs(value) > 1000.0 or abs(value) < 0.001:
            return ("%.3e" % value).replace("e", r"\times10^{") + "}"
        elif abs(value - int(value)) < 0.001:
            return "%.1f" % value
        else:
            return "%.3f" % value

    def _format_element(m, n, d):
        s = " & " if n > 0 else ""
        if type(d) == str:
            return s + d
        else:
            if abs(np.imag(d)) < settings.atol:
                return s + _format_float(np.real(d))
            elif abs(np.real(d)) < settings.atol:
                return s + _format_float(np.imag(d)) + "j"
            else:
                s_re = _format_float(np.real(d))
                s_im = _format_float(np.imag(d))
                if np.imag(d) > 0.0:
                    return s + "(" + s_re + "+" + s_im + "j)"
                else:
                    return s + "(" + s_re + s_im + "j)"

    if M > 10 and N > 10:
        # truncated matrix output
        for m in range(5):
            for n in range(5):
                s += _format_element(m, n, a[m, n])
            s += r' & \cdots'
            for n in range(N - 5, N):
                s += _format_element(m, n, a[m, n])
            s += r'\\'

        for n in range(5):
            s += _format_element(m, n, r'\vdots')
        s += r' & \ddots'
        for n in range(N - 5, N):
            s += _format_element(m, n, r'\vdots')
        s += r'\\'

        for m in range(M - 5, M):
            for n in range(5):
                s += _format_element(m, n, a[m, n])
            s += r' & \cdots'
            for n in range(N - 5, N):
                s += _format_element(m, n, a[m, n])
            s += r'\\'

    elif M > 10 and N <= 10:
        # truncated vertically elongated matrix output
        for m in range(5):
            for n in range(N):
                s += _format_element(m, n, a[m, n])
            s += r'\\'

        for n in range(N):
            s += _format_element(m, n, r'\vdots')
        s += r'\\'

        for m in range(M - 5, M):
            for n in range(N):
                s += _format_element(m, n, a[m, n])
            s += r'\\'

    elif M <= 10 and N > 10:
        # truncated horizontally elongated matrix output
        for m in range(M):
            for n in range(5):
                s += _format_element(m, n, a[m, n])
            s += r' & \cdots'
            for n in range(N - 5, N):
                s += _format_element(m, n, a[m, n])
            s += r'\\'

    else:
        # full output
        for m in range(M):
            for n in range(N):
                s += _format_element(m, n, a[m, n])
            s += r'\\'

    s += r'\end{array}\right)\end{equation*}'
    return s


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


def create_ancilla_qobj(alpha = complex(1/math.sqrt(2), 0),
                        beta = complex(1/math.sqrt(2), 0),
                        phi = np.pi/2,):
    eta = [
        [alpha**2, 0                           , 0                          ],
        [0       , beta**2/2                   , beta**2/2*cmath.exp(1j*phi)],
        [0       , beta**2/2*cmath.exp(-1j*phi), beta**2/2                  ],
    ]
    return Qobj(eta)

def evolve_ancilla_qobj(rho_st,
                        alpha = complex(1/math.sqrt(2), 0),
                        beta = complex(1/math.sqrt(2), 0),
                        phi = np.pi/2,
                        ):
    trace_factor = (p.Cp * rho_st).tr()
    a = alpha**2
    b = beta**2/2
    g2g1 = b*(np.cos(phi) - 1j*np.sin(phi)*trace_factor)
    g1g2 = b*(np.cos(phi) + 1j*np.sin(phi)*trace_factor)
    eta = [
        [a, 0        , 0        ],
        [0       , b, g2g1     ],
        [0       , g1g2     , b],
    ]
    return Qobj(eta)


def create_system_qobj(dm_type='fock', n_dims=4, **kwargs):
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
    return state
    

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


def get_last_id(parent_folder):
    """ Search the log file to get the last log ID """
    logs = pd.read_csv(f'{parent_folder}/../saved/logs.csv')
    last_id = logs['Id'].iloc[-1]
    return last_id
    
    
if __name__ == "__main__":
    print_hello()