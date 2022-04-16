import cmath, math
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

from IPython.display import Markdown, display


class QState(Qobj):
    
    def __init__(
        self, inpt, energy=1, history=None,
        dims=None, shape=None, type=None, isherm=None, copy=True, fast=False, superrep=None, isunitary=None,
        ):
        super().__init__(inpt, dims, shape, type, isherm, copy, fast, superrep, isunitary)
        # Operators
        self.dimentions = self.shape[0]
        self.am = destroy(self.dimentions)
        self.ap = create(self.dimentions)
        self._energy = energy
        self._partition = 1/self.diag()[0] if self.diag()[0] != 0 else None
        if history is None:
            self.history = []
        else:
            self.history = history
        
    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if(energy <= 0):
            raise ValueError("Energy cannot be less or equal to zero!")
        self._energy = energy

    @property
    def partition(self):
        return self._partition
        
    
    def get_temperature(self):
        """Find kT from Boltzmann distribution"""
        return - self._energy / np.log(self.diag()[1].real / self.diag()[0].real)
        
    def check_time(self, time):
        display(Markdown(f"### System State at time {time}"))
        display(self.history[time])
        sys_temp = self.history[time].get_temperature()
        display(Markdown(f"*System Temperature {'undefined' if sys_temp <0 else sys_temp}*"))
        
    def interact(self, ancilla, interaction, time):
        U = (-1j*interaction*time).expm()
        total_evolution = U * tensor(self, ancilla) * U.dag()
        self.history.append(self)
        system_evolution = QState(total_evolution.ptrace(0), 
                                  energy=self.energy,
                                  history=self.history)
        return system_evolution
    
    def meq_step(self, eta, strength, timedelta):
        first_factor = eta.plusminus * (self.ap * self * self.am - .5*commutator(self.am*self.ap, self, kind='anti'))
        second_factor =  eta.minusplus * (self.am * self * self.ap - .5*commutator(self.ap*self.am, self, kind='anti'))
        system_new = strength**2*timedelta*(first_factor + second_factor)
        self.history.append(self)
        system_new = QState(self + timedelta*system_new, 
                            energy=self.energy,
                            history=self.history)
        return system_new
        
    def is_thermal(self):
        """The Density Matrix is thermal if it is diagonal and 
        the diagonal elements are sorted from bigger to smaller"""
        in_diag = self.diag()
        out_of_diag = self - np.diag(in_diag)
        if not np.count_nonzero(out_of_diag) and np.all(np.diff(in_diag) <= 0):
            return True
        else:
            return False


class QAncilla(Qobj):
    
    def __init__(self, 
                 alpha = complex(1/math.sqrt(2), 0),
                 beta = complex(1/math.sqrt(2), 0),
                 phi = np.pi/2,
                 history = None):
        eta = [
            [alpha**2, 0                           , 0                          ],
            [0       , beta**2/2                   , beta**2/2*cmath.exp(1j*phi)],
            [0       , beta**2/2*cmath.exp(-1j*phi), beta**2/2                  ],
        ]        
        super().__init__(inpt=eta)
        self._energy = 1
        # Ancilla Operators:
        #   Sigma Plus Operator B-
        self.sigmaplus = Qobj([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        #   Sigma Minus Operator B+
        self.sigmaminus = Qobj([
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],  
        ])
        # Expectation Values
        #   B-B+ (gamma alpha)
        self.plusminus = expect(self.sigmaplus*self.sigmaminus, self)
        #   B+B- (gamma beta)
        self.minusplus = expect(self.sigmaminus*self.sigmaplus, self)
        self.factor = self.plusminus/self.minusplus
        if history is None:
            self.history = []
        else:
            self.history = history

    @property
    def energy(self):
        return self._energy
    
    def final_temperature_per_energy(self):
        return 1 / math.log(self.ratio)
    

class JointSystem(Qobj):
    def __init__(self, systems=None, n_states=2, n_dims=2, history=None):
        # Generate systems if not provided
        if systems is None:
            systems = []
            for i in range(n_states):
                system = QState(fock_dm(n_dims, 0))
                systems.append(system)
        elif not all(isinstance(system, QState) for system in systems):
            systems = [QState(system) for system in systems]
        self.systems = systems
        # Create Qobj with tensor product of the systems
        super().__init__(tensor(systems))
        
        if history is not None:
            self.history = history
        else:
            self.history = []
        # Operators
        self.ap = []
        self.am = []
        for i in range(len(systems)):
            ap = [qeye(system.dimentions) for system in systems]
            am = [qeye(system.dimentions) for system in systems]
            ap[i] = create(systems[i].dimentions)
            am[i] = destroy(systems[i].dimentions)
            self.ap.append(tensor(ap))
            self.am.append(tensor(am))
            
            
    def interact(self, ancilla, interactions, time):
        unitaries = [(-1j*interaction*time).expm() for interaction in interactions]
        self.history.append(self)
        total_system = tensor(self, ancilla)
        for U in unitaries:
            total_system = U * total_system * U.dag()
        systems = []
        for i in range(len(self.systems)):
            systems.append(total_system.ptrace(i))
        return JointSystem(systems, 
                          history=self.history)


    def __getitem__(self, index):
        return self.systems[index]
