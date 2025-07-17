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
    box_dimension:np.ndarray[float,float] = field(default_factory=lambda: np.array([10.0, 10.0]))
    n_particles:int=4
    particle_radius:float=0.5
    max_velocity:float = 0.5
    seed:int=42
    fast_initial:bool = False
    positions: Optional[np.ndarray] = None     # shape: (n_particles, 2)
    velocities: Optional[np.ndarray] = None    # shape: (n_particles, 2)
    periodic: Optional[bool] = False
    rng: np.random.Generator = field(init=False, repr=False)


    def _error_check(self):
        pass
    def __post_init__(self):
            log.info(f'creting system with {self.n_particles} particles of radii {self.particle_radius}')
            self.surface_density = (self.n_particles * np.pi * self.particle_radius**2)/(self.box_dimension[0]*self.box_dimension[1])
            log.info(f'density: {self.surface_density}, box size {self.box_dimension}')
        
            if self.surface_density > 1:
                raise ValueError("more particle that the box can pack")
            if self.surface_density > 0.8:
                log.warning("high density system. Initial state can take time to build")

            self._error_check()
            self.rng = np.random.default_rng(self.seed)
            
            if self.positions is None:
                self._set_positions()
            else:
                if self._basic_positions_check():
                    raise ValueError("inputed positions not valid. Invalid with base system configuration")
                if not self.validate_configuration(self.positions):
                    raise ValueError("inputed configuration not valid. Particles with supperpositions")
                
            self.velocities = self._set_random_velocities()
            self.i_idx, self.j_idx = np.triu_indices(self.n_particles, k=1)
            log.info(f'system created')

    def _set_positions(self):
        if self.fast_initial:
            self.positions = self._set_position_fast()
        else:
            self.positions = self._set_random_positions()

    def _set_random_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = self.particle_radius
            pos = self.rng.uniform([0,0],self.box_dimension,2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def _set_position_fast(self):
        if self.box_dimension[0]*self.box_dimension[1] < (self.particle_radius ** 2 )* self.n_particles:
            raise ValueError("can't fit the particles")
        n_boxes = (self.box_dimension // (2*self.particle_radius)).astype(int)
        pos_grid = np.zeros(n_boxes)
        positions = []
        # safe_count=0
        while len(positions) < self.n_particles:
            x,y = self.rng.integers(0,n_boxes,size=2)
            if pos_grid[x][y] == 0:
                positions.append(np.array([((2*x)+1)*self.particle_radius,((2*y)+1)*self.particle_radius]))
                pos_grid[x][y] = 1
        
        return np.array(positions,dtype=float)

    
    def _set_random_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
    def set_positions(self,positions:np.ndarray):
        valid = self.validate_configuration(new_configuration=positions)
        if (positions.shape[0]!=self.n_particles) or not valid:
            raise ValueError("new positions not valid")
        self.positions = positions


    def plot_system(self, show_velocities=False, ax=None, show_grid=False):
        if ax is None:
            _, ax = plt.subplots()

        Lx,Ly   = self.box_dimension
        R   = self.particle_radius
        pos = self.positions          # shape (N,2)

        # --- draw every disk and the images that overlap the window -------------
        shifts_x = (-Lx, 0, Lx)
        shifts_y = (-Ly, 0, Ly)           # Cartesian product gives 9 possibilities
        for p in pos:
            for dx in shifts_x:
                for dy in shifts_y:
                    # skip the central copy only once (dx=dy=0 after first loop)
                    new = p + np.array([dx, dy])

                    # Does this copy intersect the [0,L]×[0,L] window?
                    if (
                        -R <= new[0] <= Lx + R
                        and -R <= new[1] <= Ly + R
                    ):
                        ax.add_patch(plt.Circle(new, R, color="blue", alpha=0.5))

        # ------------------------------------------------------------------------
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect('equal')
        ax.set_title("Hard–disk system (periodic BCs)")
        ax.grid(show_grid)
        plt.show()

    def _basic_positions_check(self, position:Optional[np.ndarray]=None)->bool:
        if position is None:
            test_position = self.positions
        else:
            test_position = position
        if test_position.shape[0] != self.n_particles:
            raise ValueError("Invalid shape for positions")
        if test_position.dtype != np.dtype(float):
            log.warning("Given positions array is not of dtype float")
            
    def __str__(self):
        return (
            f"HardDiskSystem with {self.n_particles} particles of radius {self.particle_radius}\n"
            f"Box size: {self.box_dimension}\n"
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

    
    
