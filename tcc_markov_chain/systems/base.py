from dataclasses import dataclass, field
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

from typing import Tuple, Callable, Any, Optional, Dict, Literal
@dataclass
class DiskSystem(ABC):
    box_dimension:np.ndarray[float,float] = field(default_factory=lambda: np.array([10.0, 10.0]))
    n_particles:int=4
    particle_radius:float=0.5
    max_velocity:float = 0.5
    seed:int=42
    initial_configuration_type:Literal['random','random_sqr_lattice',"hexagonal_lattice"] = "random"
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
                self._initiate_positions()
            else:
                self.positions = np.array(self.positions,dtype=float)
                if self._basic_positions_check():
                    raise ValueError("inputed positions not valid. Invalid with base system configuration")
                if not self.validate_configuration(self.positions):
                    self.plot_system(labels=True)
                    raise ValueError("inputed configuration not valid. Particles with supperpositions")
                
            self.velocities = self._set_random_velocities()
            self.i_idx, self.j_idx = np.triu_indices(self.n_particles, k=1)
            log.info(f'system created')

    def _initiate_positions(self):
        if self.initial_configuration_type == 'random':
            self.positions = self._set_random_positions()
        if self.initial_configuration_type == "random_sqr_lattice":
            self.positions = self._random_square_lattice()
        if self.initial_configuration_type == "hexagonal_lattice":
            self.positions = self._hexagonal_lattice()

    def _hexagonal_lattice(self):
        Lx,Ly = self.box_dimension
        dx = 2*self.particle_radius
        dy = np.sqrt(3)/2 * dx
        n_cols = int(np.floor(Lx/dx))
        n_rows = int(np.floor(Ly/dy))

        if n_cols == 0 or n_rows == 0 or (n_cols*n_rows) < self.n_particles:
            log.error(f"box too small for the particles, {n_cols} cols and {n_rows} rows")
            raise ValueError("box too small")
        
        pos = []
        for r in range(n_rows):
            y=self.particle_radius+r*dy
            #offset fo even coluns
            x_offset=0.5*dx if r%2 else 0

            for c in range(n_cols):
                x = (self.particle_radius + c*dx + x_offset) % Lx
                #check for colision
                new_config = pos.copy()
                new_config.append([x,y])
                valid = self.validate_configuration(np.array(new_config,dtype=float))

                if valid:
                    pos.append([x,y])
                if len(pos)==self.n_particles:
                    return np.array(pos,dtype=float)
        # self.plot_system(labels=True)
        log.error("not enough space for the particles in the box")
        raise ValueError("not enough space for the particles in the box")
        
    def _set_random_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            if not self.periodic:
                margin = self.particle_radius
            pos = self.rng.uniform([0,0],self.box_dimension - margin,2)
            if len(positions) > 0:
                rel_pos = self.single_particle_relative_positions(pos,np.array(positions,dtype=float))
                dis = np.linalg.norm(rel_pos,axis=1)
                if  (dis > 2*self.particle_radius).all():
                    positions.append(pos)
            else:
                positions.append(pos)
        return np.array(positions)
    
    def _random_square_lattice(self):
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


    def plot_system(self, show_velocities=False, ax=None, show_grid=False,labels=False):
        if ax is None:
            _, ax = plt.subplots()

        Lx,Ly   = self.box_dimension
        R   = self.particle_radius
        pos = self.positions          # shape (N,2)

        # --- draw every disk and the images that overlap the window -------------
        shifts_x = (-Lx, 0, Lx)
        shifts_y = (-Ly, 0, Ly)           # Cartesian product gives 9 possibilities
        for i,p in enumerate(pos):
            if labels:
                plt.text(p[0],p[1],i,color='black')
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
            if test_position.shape[0] != self.n_particles:
                raise ValueError("Invalid shape for positions") 
        else:
            test_position = position
        box_a = self.box_dimension[0] * self.box_dimension[1]
        particles_a = (2 * self.particle_radius)**2 * test_position.shape[0]
        
        if test_position.dtype != np.dtype(float):
            log.warning("Given positions array is not of dtype float")
            return False
        if particles_a>box_a:
            log.warning("checked positions with invalid basic parameters")
            return False
            
    def __str__(self):
        return (
            f"HardDiskSystem with {self.n_particles} particles of radius {self.particle_radius}\n"
            f"Box size: {self.box_dimension}\n"
        )
    
    @abstractmethod
    def single_particle_relative_positions(self, particle:np.ndarray[float,float], positions:Optional[np.ndarray]=None) -> np.ndarray:
        "relative positions of a single particle"
        pass

    @abstractmethod
    def validate_configuration(self, new_configuration:np.ndarray) -> bool:
        """check if the configuration is valid"""
        pass
    
    @abstractmethod
    def calculate_relative_positions(self, positions:Optional[np.ndarray] = None)->np.ndarray:
        pass

    @abstractmethod
    def update_particle_position(self, k:int,displacement:np.ndarray) -> np.ndarray:
        """
        Change a single particle position in the system based on the displacement.
        It does not update the system configuration.
        Return new_configuration (valid or invalid)
        """
        pass

    
    
