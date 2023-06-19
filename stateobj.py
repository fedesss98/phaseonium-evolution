import cmath, math
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

from IPython.display import Markdown, display


class QState(Qobj):
    
    def __init__(
        self, inpt, energy=1, history=None, n_subsys=None,
        dims=None, shape=None, type=None, isherm=None, copy=True, fast=False, superrep=None, isunitary=None,
        ):
        super().__init__(inpt, dims, shape, type, isherm, copy, fast, superrep, isunitary)
        # Operators
        self.dimentions = self.shape[0]
        self.am = destroy(self.dimentions)
        self.ap = create(self.dimentions)
        self._energy = energy
        self.n_subsys = n_subsys
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
    
    def interact_multiple(self, ancilla, interactions, time):
        unitaries = [(-1j*interaction*time).expm() for interaction in interactions]
        self.history.append(self)
        total_system = tensor(self, ancilla)
        for U in unitaries:
            total_system = U * total_system * U.dag()
        # Trace off Ancillas
        system_evolution = QState(total_system.ptrace(range(self.n_subsys)), 
                                  history=self.history,
                                  n_subsys=self.n_subsys)
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
                 eta = None,
                 alpha = complex(1/math.sqrt(2), 0),
                 beta = complex(1/math.sqrt(2), 0),
                 phi = np.pi/2,
                 history = None):
        if eta is None:
            eta = [[alpha**2, 0                           , 0                          ],
                   [0       , beta**2/2                   , beta**2/2*cmath.exp(1j*phi)],
                   [0       , beta**2/2*cmath.exp(-1j*phi), beta**2/2                  ],]
            self._alpha = alpha
            self._beta = beta
            self._phi = phi
        else:
            alpha = cmath.sqrt(eta.full()[0, 0])
            beta = cmath.sqrt(2*eta.full()[1, 1])
            phi = math.acos((eta.full()[1, 2]/eta.full()[1, 1]).real)
            self._alpha = alpha
            self._beta = beta
            self._phi = phi
        
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
        #   <B-B+> (gamma alpha)
        self.plusminus = expect(self.sigmaplus*self.sigmaminus, self)
        #   <B+B-> (gamma beta)
        self.minusplus = expect(self.sigmaminus*self.sigmaplus, self)
        #   <[B+ ; B-]>
        self.commutator = expect(commutator(self.sigmaminus, self.sigmaplus), self)
        # Ancilla factor (predicts stable temperature)
        self.factor = self.plusminus/self.minusplus
        if history is None:
            self.history = []
        else:
            self.history = history

    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def phi(self):
        return self._phi
        
    @property
    def energy(self):
        return self._energy
    
    def final_temperature_per_energy(self):
        return 1 / math.log(self.factor)
    

class JointSystem(Qobj):
    def __init__(self, systems=None, n_states=2, n_dims=2, history=None):
        # Generate systems if not provided
        if systems is None:
            systems = []
            for i in range(n_states):
                system = QState(fock_dm(n_dims, 0))
                systems.append(system)
            system = tensor(systems)
        elif len(systems)==1 or isinstance(systems, QState):
            system = systems.pop()
            for n in range(n_states):
                systems.append(QState(system.ptrace([n])))
        elif not all(isinstance(system, QState) for system in systems):
            systems = [QState(system) for system in systems]
            system = tensor(systems)
        else:
            system = tensor(systems)
        self.systems = systems
        # Create Qobj with tensor product of the systems
        super().__init__(system)
        
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
        self.total_ap = sum(self.ap)
        self.total_am = sum(self.am)
        
    
    def interact(self, ancilla, interactions, time, historical=True):
        """
        Unitary Evolution.
        Parameters
        ----------
        ancilla: QAncilla (Qobj)
        interactions: array or list
        time: float
        """
        if historical:
            self.history.append(self)
        unitaries = [(-1j*interaction*time).expm() for interaction in interactions]
        total_system = tensor(self, ancilla)
        for U in unitaries:
            total_system = U * total_system * U.dag()  # U[1]*(U[0]*tot*U[0].dag())*U[1].dag()
        # Trace-off the Ancilla
        joint_system = total_system.ptrace(range(len(self.systems)))  # ptrace([0, 1])
        return JointSystem([joint_system], 
                           history=self.history,
                           n_states=len(self.systems))
    
    def meq_step(self, ancilla, strenght, timedelta, historical=True):
        """
        Master Equation Evolution.
        Parameters
        ----------
        ancilla: QAncilla (Qobj)
        strenght: float, strenght of interaction
        timedelta: float
        """
        if historical:
            self.history.append(self)
        # First Term (commutator)
        hamilton_prime = expect(ancilla.sigmaplus, ancilla)*self.total_am
        hamilton_prime += expect(ancilla.sigmaminus, ancilla)*self.total_ap
        hamilton_prime *= strenght
        
        hamilton_second = expect(commutator(ancilla.sigmaplus, ancilla.sigmaminus), ancilla)*self.am[0]*self.ap[1]
        hamilton_second += expect(commutator(ancilla.sigmaminus, ancilla.sigmaplus), ancilla)*self.am[1]*self.ap[0]
        hamilton_second *= 1j*timedelta/4*strenght**2
        
        comm = commutator(hamilton_prime+hamilton_second, self)
        # Second Term (Dissipator)
        first_factor = self.total_ap * self * self.total_am
        first_factor -= 0.5*commutator(self.total_am*self.total_ap, self, kind='anti')
        first_factor *= ancilla.plusminus
        
        second_factor =  self.total_am * self * self.total_ap 
        second_factor -= 0.5*commutator(self.total_ap*self.total_am, self, kind='anti')
        second_factor *= ancilla.minusplus
        
        diss = strenght**2*timedelta*(first_factor + second_factor)
        
        meq = -1j/2*comm + 1/4*diss
        system_new = self + timedelta*meq
        return JointSystem([system_new], 
                           history=self.history,
                           n_states=len(self.systems))


    def __getitem__(self, index):
        return self.systems[index]


class Physics:
    def __init__(self, dimension, interaction_time, interaction_strength):
        theta = 1 * interaction_strength * interaction_time
        # Creation and Annihilation Operators
        self.ad = create(dimension)
        self.a = destroy(dimension)
        
        self.a1 = tensor(self.a, qeye(dimension))
        self.ad1 = tensor(self.ad, qeye(dimension))
        self.a2 = tensor(qeye(dimension), self.a)
        self.ad2 = tensor(qeye(dimension), self.ad)
        # Number Operators
        self.ada = self.ad * self.a
        self.aad = self.ad * self.a + 1
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

        # Bosonic Operators
        self.C = (theta*(2*self.aad).sqrtm()).cosm()
        self.Cp = (theta*(2*self.ada).sqrtm()).cosm()
        dividend = ((2*self.aad).sqrtm()).inv()
        sine = (theta*(2*self.aad).sqrtm()).sinm()
        self.S = self.ad * sine * dividend
        self.Sd = sine * dividend * self.a
        
        # Interaction
        self.V = tensor(self.a, self.sigmaplus) + tensor(self.ad, self.sigmaminus)
        # Entangled System interactions
        self.V1 = tensor(self.a1, self.sigmaplus) + tensor(self.ad1, self.sigmaminus)
        self.V2 = tensor(self.a2, self.sigmaplus) + tensor(self.ad2, self.sigmaminus)