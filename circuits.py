"""
Circuit and Quantum Gate utilities

"""
import qutip
import numpy as np
import random


class Gate:
    def __init__(self, controller, lanes: tuple | list, name='', control_lane=None):
        self.controller = controller

        self.lanes = list(lanes)
        self.name = name
        self.control_lane = control_lane
        self.controlled = True if control_lane is not None else False

    @property
    def used_systems(self):
        lanes = [self.controller.lanes[lane] for lane in self.lanes]
        systems = []
        duplicated_systems = []
        for lane in lanes:
            if lane.system is not None and not lane.is_entangled():
                systems.append(lane.system)
            else:
                # Append only lanes whose systems are not already in the list
                if lane.system not in duplicated_systems:
                    systems.append(lane.system)
                    duplicated_systems.append(lane.system)
        return systems

    @staticmethod
    def check(control_system):
        if control_system == -1:
            return False
        else:
            return True

    def evolve(self, system):
        return system

    def inject(self, step, system):
        next_step = step + 1
        for lane in self.lanes:
            series = self.controller.lanes[lane].system_evolution
            series[step + 1: ] = [system for _ in range(len(series) - next_step)]
        return None

    def __str__(self):
        return f"{self.name} acting on {self.lanes}"


class Lane:
    def __init__(self, controller, system: qutip.Qobj | int | None = None, name=''):
        self.controller = controller
        self.name = name
        self._system_evolution = [system for _ in range(len(self.controller.gates) + 1)]

    @property
    def system_evolution(self):
        return self._system_evolution

    @system_evolution.setter
    def system_evolution(self, value):
        self._system_evolution = value

    @property
    def system(self) -> qutip.Qobj | int | None:
        """Return current step system"""
        return self._system_evolution[self.controller.step]
    
    @system.setter
    def system(self, value):
        if not isinstance(value, qutip.Qobj) and value is not None:
            if isinstance(value, int):
                if value not in [1, -1]:
                    raise ValueError("Classical systems may only take +1 or -1 values")
            else:
                raise ValueError("System in the circuit may only be qutip.Qobj objects or +1 or -1.")
        self._system_evolution[self.controller.step] = value
        
    def is_entangled(self):
        if isinstance(self.system, int):
            return False
        else:
            rows = self.system.dims[0]
            columns = self.system.dims[1]
        if len(rows) == 1 and len(columns) == 1:
            return False
        else:
            return True

    def __str__(self):
        if isinstance(self.system, qutip.Qobj):
            system_string = f"Quantum System {self.system.dims[0]}x{self.system.dims[1]} "
            system_string += f"({'entangled' if self.is_entangled() else 'not entangled'})"
        else:
            system_string = f"Classical System - value: {self.system}"
        return f"{self.name}: {system_string}"

    def __repr__(self):
        if isinstance(self.system, qutip.Qobj):
            system_string = f"Quantum System {self.system.dims[0]}x{self.system.dims[1]} "
            system_string += f"({'entangled' if self.is_entangled() else 'not entangled'})"
        else:
            system_string = f"Classical System - value: {self.system}"
        return f"{self.name}: {system_string}"


class Circuit:
    def __init__(self, n_lanes=1, logging=False):
        self.gates = []
        self.lanes = [Lane(self) for _ in range(n_lanes)]
        self.gate_map = {}
        self.step = 0
        self.logging: bool = logging
        self.results: dict = {}
        self.t = 0

    @property
    def matrix(self):
        rows = len(self.lanes)
        columns = len(self.gates)
        matrix = [[0] for row in range(rows)]
        for row in range(rows):
            for col in range(columns):
                if self.gates[col].control_lane == row:
                    code = self.lanes[row].system
                elif row in self.gates[col].lanes:
                    code = self.gates[col].name
                else:
                    code = 0
                matrix[row].append(code)
        return matrix
        
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
        if self.logging:
            print(f"Removed {gate.name} gate")
        self._update_matrix(position, gate, '-')
        return None

    @staticmethod
    def _select_systems(step, used_lanes):
        used_systems = []  # Used lanes without duplicates
        duplicated_systems = []  # Systems that appear more than once
        for lane in used_lanes:
            current_system = lane.system_evolution[step + 1]
            if not lane.is_entangled():
                used_systems.append(current_system)
            else:
                # Append only lanes whose systems are not already in the list
                if current_system not in duplicated_systems:
                    duplicated_systems.append(current_system)
                    used_systems.append(current_system)
        return used_systems

    def evaluate_gate(self, gate, step):
        try:
            # Entangle systems
            total_system = qutip.tensor(gate.used_systems)
        except TypeError:
            total_system = gate.used_systems
        if self.logging:
            print(f"Step {step} - Gate {gate}")
            print(f"Used lanes: {[self.lanes[i] for i in gate.lanes]}")
            print(f"{'_' * 40}\n")
        if not gate.controlled:
            total_system = gate.evolve(total_system)
            gate.inject(step, total_system)
        else:
            control_system = self.lanes[gate.control_lane].system
            if gate.check(control_system):
                total_system = gate.evolve(total_system)
                gate.inject(step, total_system)
        return total_system

    def run(self):
        for step, gate in enumerate(self.gates):
            self.step = step
            self.evaluate_gate(gate, step)
        return None

    def loop(self, times, start=0, stop=-1):
        stop = len(self.gates) if stop == -1 else stop
        loop_gates = self.gates[start: stop]
        for t in range(times):
            self.t += 1
            for step, gate in enumerate(loop_gates):
                self.step = step + start
                self.evaluate_gate(gate, step)
            # Replace starting systems with evolved ones
            for lane in self.lanes:
                evolved_system = lane.system_evolution[self.step + 1]
                lane.system_evolution[start:] = [evolved_system for _ in range(len(self.gates) + 1 - start)]
        return None

    def draw(self):
        columns = len(self.gates) + 1
        ascii_string = ''
        max_header_len = max([len(lane.name) for lane in self.lanes])
        max_name_len = max([len(gate.name) for gate in self.gates])
        max_name_len = max_name_len if max_name_len % 2 == 0 else max_name_len + 1
        ascii_string += f'{" "*(max_header_len + max_name_len + 3)}'
        # Print column numbers
        for n_gate in range(1, columns):
            ascii_string += f'{n_gate:^{max_name_len + 2}}'
        ascii_string += '\n'
        for row, lane in enumerate(self.lanes):
            ascii_string += f'{lane.name:<{max_header_len}} '
            for col in range(columns):
                if self.matrix[row][col] == 0:
                    ascii_string += f'-{"-"*(max_name_len)}-'
                else:
                    matrix_code = self.matrix[row][col]
                    if matrix_code is not None:
                        gate_name = '||' if matrix_code == -1 else '==' if matrix_code == 1 else matrix_code
                    else:
                        gate_name = '  '
                    ascii_string += f'-{gate_name:^{max_name_len}}-'
            ascii_string += '---\n' if lane.system_evolution[-1] is not None else '   \n'
        print(ascii_string)
        return None

    def __getitem__(self, item):
        return self.lanes[item].system
