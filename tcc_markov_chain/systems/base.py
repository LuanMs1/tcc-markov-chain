from dataclasses import dataclass, field
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

from typing import Tuple, Callable, Any, Optional, Dict
@dataclass
class DiskSystem(ABC):
    box_size:float=10
    n_particles:int=4
    particle_radius:float=0.5
    max_velocity:float = 0.5
    seed:int=42
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)

    def _error_check(self):
        pass
    def __post_init__(self):
            log.info(f'creting system with {self.n_particles} particles of radii {self.particle_radius}')
            self.surface_density = (self.n_particles * np.pi * self.particle_radius**2)/self.box_size**2
            log.info(f'density: {self.surface_density}, box size {self.box_size}')
        
            if self.surface_density > 1:
                raise ValueError("more particle that the box can pack")
            if self.surface_density > 0.8:
                log.warning("high density system. Initial state can take time to build")

            self._error_check()
            self.rng = np.random.default_rng(self.seed)
            self.positions = self._set_random_positions()
            self.velocities = self._set_random_velocities()
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
    
    def _set_random_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
    def set_positions(self,positions:np.ndarray):
        valid = self.validate_configuration(new_configuration=positions)
        if (positions.shape[0]!=self.n_particles) or not valid:
            raise ValueError("new positions not valid")
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

    def _basic_positions_check(self, position:Optional[np.ndarray]=None)->bool:
        if isinstance(position,np.ndarray):
            test_position = position
        else:
            test_position = self.positions
        if test_position.shape[0] != self.n_particles:
            raise ValueError("Invalid shape for positions")
        if test_position.dtype != np.dtype(float):
            log.warning("Given positions array is not of dtype float")
            
    def __str__(self):
        return (
            f"HardDiskSystem with {self.n_particles} particles of radius {self.particle_radius}\n"
            f"Box size: {self.box_size}\n"
        )
    
    @abstractmethod
    def validate_configuration(self, new_configuration:np.ndarray) -> bool:
        """check if the configuration is valid"""
        pass
    
    @abstractmethod
    def calculate_relative_positions(self, positions:Optional[np.ndarray] = None)->np.ndarray:
        pass

    def update_particle_position(self, k:int,displacement:np.ndarray) -> np.ndarray:
        """
        Change a single particle position in the system based on the displacement.
        It does not update the system configuration.
        Return new_configuration (valid or invalid)
        """
        pass

    
    
