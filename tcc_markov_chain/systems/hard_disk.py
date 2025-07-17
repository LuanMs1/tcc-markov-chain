from .base import DiskSystem
from dataclasses import dataclass
import numpy as np
from ..utils.logging import setup_logging,get_logger
setup_logging()
log = get_logger(__name__)

from typing import Tuple, Callable, Any, Optional, Dict

@dataclass
class HDPeriodic(DiskSystem):
    def __post_init__(self):
        super().__post_init__()
        self.periodic=True
    
    def single_particle_relative_positions(self, particle, positions=None) ->np.ndarray:
        if positions is None:
            positions = self.positions
        
        rel_pos = positions - particle
        rel_pos -= self.box_dimension *np.rint(rel_pos/self.box_dimension) 

        return rel_pos

    def calculate_relative_positions(self, positions:Optional[np.ndarray] = None):
        if isinstance(positions,np.ndarray):
            i_idx, j_idx = np.triu_indices(positions.shape[0],k=1)
        else:
            positions=self.positions.copy()
            i_idx = self.i_idx
            j_idx = self.j_idx
    
        dis = positions[i_idx] - positions[j_idx]
        dis -= self.box_dimension * np.rint(dis/self.box_dimension)
        return dis

    def validate_configuration(self, new_configuration:np.ndarray):
        if (new_configuration < 0).any() or (new_configuration > self.box_dimension).any():
            log.error("Particle out of bound")
            return False
        dx = self.calculate_relative_positions(new_configuration)
        distances = np.linalg.norm(dx,axis=1)
        #-1e-14 for a float point tolerance
        if (distances < (2*self.particle_radius)-1e-14).any():
            return False
        else:
            return True
    
    def move_particle(self, particle:np.ndarray[float,float],displacement:float) -> np.ndarray[float,float]:
        return (particle+ displacement) % self.box_dimension
    
    def update_particle_position(self, k, displacement):
        if k>self.n_particles-1:
            raise ValueError("Invalid particle")
        if displacement.shape!=self.positions[0].shape:
            raise ValueError("Wrong displacement dimension")
        
        new_particle_position = self.move_particle(self.positions[k],displacement)
        others = self.positions[[i for i in range(self.n_particles) if i!=k]]

        #relative position compared to others of new position
        rel_pos = self.single_particle_relative_positions(new_particle_position,others)
        # rel_pos = self.positions[[i for i in range(self.n_particles) if i!=k]] - new_particle_position
        # rel_pos -= self.box_size * np.rint(rel_pos/self.box_size)
        dis = np.linalg.norm(rel_pos,axis=1)
        #reject
        if (dis<2*self.particle_radius).any():
            return False #to say it was rejected
        
        self.positions[k] = new_particle_position
        return True #indication that was accepted