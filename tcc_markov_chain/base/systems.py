from dataclasses import dataclass, field
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from abc import abstractmethod, ABC

class ParticleSystem(ABC):
    @property
    @abstractmethod
    def box_size(self) -> float:
        pass

    @property
    @abstractmethod
    def particle_radius(self) -> float:
        pass


@dataclass
class HardDiskSystem():
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    seed: int = 42
    max_velocity:int = 1
    periodic_boundary:bool = False
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)

    def _error_check(self):
        if len(self.positions.shape) != 2:
            raise ValueError("array of positions needs to have 2 dimensions")
        if len(self.velocities.shape) != 2:
            raise ValueError("array of velocities needs to have 2 dimensions")
        if 2*self.particle_radius > self.box_size:
            raise ValueError("particle radius dyameter needs to be less them box_size")
        
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_positions()
        self.velocities = self._set_velocities()
        self._error_check()


    def _set_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = (not self.periodic_boundary) * self.particle_radius
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
        if system.periodic_boundary:
            raise TypeError('the system must have aperiodic boundary')
        self.system = system
        self.time = 0.

    # def particle_wall_colision(sel)
    def _next_wall_colision(self) -> Tuple[np.array, np.array]:
        """
            calculate the wall colisions for each particle.
            return the first wall that each particle will colide
        """
        pos_mask = self.system.velocities > 0
        neg_mask = self.system.velocities < 0

        #set the direction mask
        t_plus = np.full_like(self.system.positions, np.inf)
        t_neg = np.full_like(self.system.positions, np.inf)

        #calculate the time of colision in each direction
        pos_dis = self.system.box_size - self.system.particle_radius - self.system.positions[pos_mask]
        t_plus[pos_mask] = pos_dis / self.system.velocities[pos_mask]
        neg_dis = self.system.positions[neg_mask] - self.system.particle_radius
        t_neg[neg_mask] = neg_dis / -self.system.velocities[neg_mask]


        #set the minimun colision time in each dimension for the particles
        times = np.minimum(t_neg, t_plus)
        #wall that was hitted (0 for colision in x and 1 for colision in y)
        wall_hit = np.argmin(times,axis=1)
        hit_times = times[np.arange(self.system.n_particles), wall_hit]

        return wall_hit, np.argmin(hit_times), min(hit_times)
    
    def _next_pair_colision(self):
        return 0, np.inf

    def _update_position(self, dt):
        self.system.positions = self.system.positions + (self.system.velocities * dt)
        
    def _pair_colision(self, i, j):
        '''update particle i,j velocities'''
        pass

    def next_colision(self):
        axis_index, wall_particle, wall_time = self._next_wall_colision()
        particle_pair, pair_time = self._next_pair_colision()

        next_colision_time = min(wall_time, pair_time)
        self._update_position(next_colision_time - self.time)

        if wall_time < pair_time:
            self.system.velocities[wall_particle][axis_index] *= -1
        else:
            self._pair_colision(particle_pair[0], particle_pair[1])

    def step(self):
        """simulate one step"""
        self.next_colision()
    def run(self, n_steps = 5):
        '''simulate all steps'''
        for i in range(n_steps):
            self.system.plot_system()
            self.next_colision()