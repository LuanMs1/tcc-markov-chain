from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, Any
from abc import ABC,abstractmethod
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)


@dataclass
class HardDiskSystem(ABC):
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    seed: int = 42
    max_velocity:int = 1
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    # velocities:np.ndarray = field(init=False)

    def _error_check(self):
        # if len(self.positions.shape) != 2:
        #     raise ValueError("array of positions needs to have 2 dimensions")
        # if len(self.velocities.shape) != 2:
        #     raise ValueError("array of velocities needs to have 2 dimensions")
        if 2*self.particle_radius > self.box_size:
            raise ValueError("particle radius dyameter needs to be less them box_size")
        
    def __post_init__(self):
        log.info(f'creting system with {self.n_particles} particles of radii {self.particle_radius} and box size {self.box_size}')
        surface_density = (self.n_particles * np.pi * self.particle_radius**2)/self.box_size**2
        log.info(f'density: {surface_density}')
        self._error_check()
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_random_positions()
        # self.velocities = self._set_random_velocities()
        self.initial_positions = self.positions.copy()
        # self.initial_velocities = self.velocities.copy()
        self.i_idx, self.j_idx = np.triu_indices(self.n_particles, k=1)
        log.info(f'system created')

    def _set_random_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = self.particle_radius
            pos = self.rng.uniform(0 + margin, self.box_size - margin, size = 2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def set_positions(self,positions:np.ndarray):
        if (positions.shape[0]!=self.n_particles):
            raise ValueError("positions needs to have same size of the system")

        self.positions = positions
        
        
    def plot_system(self, show_velocities=False, ax = None, show_grid = False):
        if ax == None:
            fig, ax = plt.subplots()    
        
        # Plotando os discos
        for pos in self.positions:
            circle = plt.Circle(pos, self.particle_radius, color='blue', alpha=0.5)
            ax.add_patch(circle)
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_aspect('equal')
        ax.set_title("Hard Disk System")
        plt.grid(show_grid)
    