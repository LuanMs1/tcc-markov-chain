from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, Any
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)

#Performance and modularization improvements
#1. Update next events only to particles involved in the events
#2. Change the NxNxD array of relative posistions to N(N-1)/2,D using np.triu_indices(n,k=1) 
#3. do the same of 2 for every calculation. Without this, everything will brake
@dataclass
class Event:
    time:float
    trigger:Callable[...,None]
    func_args:Dict[str,Any]

    def call_trigger(self):
        self.trigger(**self.func_args)

@dataclass
class ParticleSystem(ABC):
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    max_velocity:float=1.
    seed: int = 42
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)

    def _error_check(self):
        # if len(self.positions.shape) != 2:
        #     raise ValueError("array of positions needs to have 2 dimensions")
        # if len(self.velocities.shape) != 2:
        #     raise ValueError("array of velocities needs to have 2 dimensions")
        density = ((self.particle_radius**2) * np.pi * self.n_particles) / (self.box_size**2)
        if density>0.6:
            log.warning("Packad system. Creating a sample can take time")
        if density>1:
            raise ValueError("particles can't feat the box")
        
    def __post_init__(self):
        self._error_check()
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_positions()
        self.velocities = self._set_velocities()
        self.initial_positions = self.positions.copy()
        self.initial_velocities = self.velocities.copy()
        log.info(f'system created with {self.n_particles} particles')
        log.info("calculating next events")
        # self._calculate_events()
    
    def _set_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = self.particle_radius
            pos = self.rng.uniform(0 + margin, self.box_size - margin, size = 2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def _set_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
    def get_relative_positions(self) -> np.ndarray:
        """
        get the distances between particle pairs 
        Returns:
            distances (np.ndarray): pairwise relative positions.
        """
        # particle_pairs (List[Tuple[int, int]]): List of (i, j) index pairs corresponding to each distance.

        relative_positions = self.positions[:,np.newaxis,:] - self.positions[np.newaxis,:,:]
        return relative_positions
    
    def get_relative_velocities(self) -> np.ndarray:
        relative_velocities = self.velocities[:,np.newaxis,:] - self.velocities[np.newaxis,:,:]
        return relative_velocities
    
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
        plt.show()

    # @abstractmethod
    # def _update_position(self, dt:float):
    #     """update position after an dt interval"""
    #     pass    
    # @abstractmethod
    # def _sp_next_event(self) -> Event:
    #     """Calculate the next event for a single particle"""
    #     pass    
    # @abstractmethod
    # def _get_next_event(self):
    #     """Get the next event and set a Event object"""
    #     pass
    # @abstractmethod
    # def _calculate_events(self):
    #     """define the next event for every particle and set as attribute"""
    #     pass

@dataclass
class HardDisckWalls(ParticleSystem):
    
    def _single_particle_next_colisions(self, i:int):
        pass

    def _next_pair_collision(self):
        rel_dis = self.get_relative_positions()
        rel_vel = self.get_relative_velocities()

        dx_dv = np.einsum("ijk,ijk->ij",rel_dis,rel_vel)
        dx_dx = np.einsum("ijk,ijk->ij",rel_dis,rel_dis)
        dv_dv = np.einsum("ijk,ijk->ij",rel_vel,rel_vel)
        
        delta = np.square(dx_dv) - dv_dv * (dx_dx - 4*self.particle_radius**2)
        mask = (delta>-1e-12) & (dx_dv<0)
        np.fill_diagonal(mask, False)

        time_to_colide = np.full_like(delta, np.inf, dtype=float)
        time_to_colide[mask] = - (dx_dv[mask] + np.sqrt(delta[mask])) / dv_dv[mask]

        i,j = np.where(time_to_colide == time_to_colide.min())

        return (i[0], j[0]), time_to_colide[i[0], j[0]]
        # self.pair_collision_times = time_to_colide
        # return time_to_colide

    def _pair_collision(self, i, j):
        '''update particle i,j velocities'''
        #calcular vetor normal ao plano de colisao
        dx = self.positions[i] - self.positions[j]
        dv = self.velocities[i] - self.velocities[j]

        collision_normal = dx / np.sqrt(np.dot(dx,dx))
        magnitude_change = np.dot(collision_normal,dv)
        
        self.velocities[i] = self.velocities[i] - magnitude_change * collision_normal
        self.velocities[j] = self.velocities[j] + magnitude_change * collision_normal

    def _update_posistion(self,dt:float):
        self.positions = self.positions + (self.velocities * dt)