"""
Circuit and Quantum Gate utilities

"""
import qutip
import numpy as np
import random

class Gate:
    def __init__(self, lanes: tuple | list, name='', control_lane=None):
        self.lanes = list(lanes)
        self.name = name
        self.control_lane = control_lane
        self.controlled = True if control_lane is not None else False
        
    def check(self, control_system):
        if control_system == -1:
            return False
        else:
            return True
        
    def evolve(self, system):
        return system


class Lane:
    def __init__(self, system: qutip.Qobj | None = None, name=''):
        self.system = system
        self.name = name
        
    @property
    def system(self):
        return self._system
    
    @system.setter
    def system(self, value):
        if not isinstance(value, qutip.Qobj) and value is not None:
            if isinstance(value, int):
                if value not in [1, -1]:
                    raise ValueError("Classical systems may only take +1 or -1 values")
            else:
                raise ValueError("System in the circuit may only be qutip.Qobj objects or +1 or -1.")
        self._system = value
        
    def is_entangled(self):
        rows = self.system.dims[0]
        columns = self.system.dims[1]
        if len(rows) == 1 and len(columns) == 1:
            return True
        else:
            return False



class Circuit:
    def __init__(self, n_lanes=1):
        self.lanes = [Lane() for lane in range(n_lanes)]
        self.gates = []
        # Representation Matrix
        self.matrix = [[0] for i in range(n_lanes)]
        self.gate_map = {}
        
    def _update_matrix(self, gate_position, gate, method='+'):
        columns = len(self.gates) + 1
        rows = len(self.lanes)
        # self.matrix = [[0 for col in range(columns)] for row in range(rows)]
        for row in range(rows):
            if method == '+':
                if row in gate.lanes:
                    code = gate.name
                elif row == gate.control_lane:
                    code = self.lanes[gate.control_lane].system
                else:
                    code = 0
                self.matrix[row].insert(gate_position + 1, code)
            elif method == '-':
                self.matrix[row].pop(gate_position + 1)    
        return None

    def _update_lane(self, lane, system):
        # Search the position of the controlled gate if any
        for col, gate in enumerate(self.gates):
            if gate.control_lane == lane:
                self.matrix[lane][col+1] = system 
        return None
    
    def populate_lane(self, lane, system, name='undefined'):
        self.lanes[lane].system = system
        self.lanes[lane].name = name
        
    def update_lane(self, lane, system=None, name=None):
        if system is not None:
            self.lanes[lane].system = system
            # Check if it's a classical bit and control something
            if not isinstance(system, qutip.Qobj):
                self._update_lane(lane, system)
        if name is not None:
            self.lanes[lane].name = name
        return None

    def add_gate(self, gate: Gate, position=0):
        gate_code = random.randint(100, 999)
        self.gates.insert(position, gate)
        self.gate_map.update({gate.name: position})
        print(f"Created {gate.name} gate with code {gate_code}")
        self._update_matrix(position, gate)
        return None
    
    def remove_gate(self, position=0):
        gate = self.gates.pop(position)
        print(f"Removed {gate.name} gate")
        self._update_matrix(position, gate, '-')
        return None
        
        
    def _select_systems(self, used_lanes):
        entangled_lanes = []
        unique_entangled_lanes = []
        pure_lanes = []
        for lane in used_lanes:
            if lane.is_entangled():
                entangled_lanes.append(lane)
            else:
                pure_lanes.append(lane)
        for lane in entangled_lanes:
            if lane not in unique_entangled_lanes:
                unique_entangled_lanes.append(lane)
        systems = [lane.system for lane in pure_lanes + unique_entangled_lanes]
        return systems
        
    def run(self):
        for gate in self.gates:
            used_lanes = [self.lanes[i] for i in gate.lanes]
            systems = self._select_systems(used_lanes)
            # Entangle systems
            total_system = qutip.tensor(systems)
            if not gate.controlled:
                total_system = gate.evolve(total_system)
            else:
                control_system = self.lanes[gate.control_lane].system
                if gate.check(control_system):
                    total_system = gate.evolve(total_system)
            # Entangled system are inkected in every lane
            for i in range(len(used_lanes)):
                used_lanes[i].system = total_system
        return None
        
    def draw(self):
        columns = len(self.gates) + 1
        ascii_string = ''
        max_header_len = max([len(lane.name) for lane in self.lanes])
        max_name_len = max([len(gate.name) for gate in self.gates])
        max_name_len = max_name_len + 1 if max_name_len % 2 == 0 else max_name_len
        ascii_string += f'{" "*(max_header_len + max_name_len + 2)}'
        for n_gate in range(1, columns):
            ascii_string += f'{n_gate:^{max_name_len + 2}}'
        ascii_string += '\n'
        for row, lane in enumerate(self.lanes):
            ascii_string += f'{lane.name:<{max_header_len}} '
            for col in range(columns):
                if self.matrix[row][col] == 0:
                    ascii_string += f'{"-"*(max_name_len + 1)} '
                else:
                    matrix_code = self.matrix[row][col]
                    gate_name = 'x' if matrix_code == -1 else 'O' if matrix_code == 1 else matrix_code
                    ascii_string += f' {gate_name:^{max_name_len}} '
            ascii_string += '---\n'
        print(ascii_string)
        return None