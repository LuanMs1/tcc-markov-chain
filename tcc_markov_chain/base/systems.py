from dataclasses import dataclass, field
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

    
@dataclass
class HardDiskSystem():
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    seed: int = 42
    max_velocity:int = 1
    periodict_boundary:bool = field(init=False, default=False)
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_positions()
        self.velocities = self._set_velocities()

    def _set_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = self.periodict_boundary * self.particle_radius
            pos = self.rng.uniform(0 + margin, self.box_size - margin, size = 2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def _set_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
    def plot_system(self, show_velocities=True, ax = None, show_grid = False):
        if ax == None:
            fig, ax = plt.subplots()    
        
        # Plotando os discos
        for pos in self.positions:
            circle = plt.Circle(pos, self.particle_radius, color='blue', alpha=0.5)
            ax.add_patch(circle)
        
        # Plotando as velocidades como setas
        if show_velocities:
            ax.quiver(
                self.positions[:, 0], self.positions[:, 1],  # x, y das partículas
                self.velocities[:, 0], self.velocities[:, 1],  # componentes vx, vy
                color='black', angles='xy', scale_units='xy', scale=1, width=0.005
            )
        
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_aspect('equal')
        ax.set_title("Hard Disk System")
        plt.grid(show_grid)
        # plt.show()

class BoundarySimulation():

    def __init__(self, system:HardDiskSystem, n_steps:int= 5):
        if system.periodict_boundary:
            raise TypeError('the system must have aperiodic boundary')
        self.system = system
        self.time = 0.
        self.n_steps = n_steps

    def particle_wall_colision(sel)
    def _next_wall_colision(self) -> Tuple[float, int]:
        times = []
        walls = []
        for d in range(2):


    def _next_pair_colision(self) -> Tuple[float, Tuple[np.array, np.array]]:
        pass

    def _update_position(self, dt):
        self.system.positions = self.system.positions * (self.system.velocities * dt)

    def _wall_colision(self, vel, wall_index):
        '''update velocitie in wall colision'''
        if wall_index in (0, 2):
            vel[1] *= -1
        if wall_index in (1, 3):
            vel[0] *= -1

        
    def _pair_colision(self, i, j):
        '''update particle i,j velocities'''
        pass

    def next_colision(self):
        wall_time, wall_disk = self._next_wall_colision()
        pair_time, pair_colision = self._next_pair_colision()

        next_colision_time = min(wall_time, pair_time)
        self._update_position(next_colision_time - self.time)

        if wall_time < pair_time:
            self._wall_colision(wall_disk)
        else:
            self._pair_colision(pair_colision[0], pair_colision[1])

    def step(self):
        """simulate one step"""
        pass

    def run(self):
        '''simulate all steps'''