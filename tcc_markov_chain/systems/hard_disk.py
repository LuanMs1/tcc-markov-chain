from .base import DiskSystem
from dataclasses import dataclass
import numpy as np


from typing import Tuple, Callable, Any, Optional, Dict

@dataclass
class HDPeriodic(DiskSystem):
    def __post_init__(self):
        super().__post_init__()
        self.periodic=True
    
    def calculate_relative_positions(self, positions:Optional[np.ndarray] = None):
        if isinstance(positions,np.ndarray):
            i_idx, j_idx = np.triu_indices(positions.shape[0],k=1)
        else:
            positions=self.positions.copy()
            i_idx = self.i_idx
            j_idx = self.j_idx
    
        dis = positions[i_idx] - positions[j_idx]
        dis -= self.box_size * np.rint(dis/self.box_size)
        return dis

    def validate_configuration(self, new_configuration:np.ndarray):
        dx = self.calculate_relative_positions(new_configuration)
        distances = np.linalg.norm(dx,axis=1)
        if (distances < 2*self.particle_radius).any():
            return False
        else:
            return True
        
    def update_particle_position(self, k, displacement):
        if k>self.n_particles-1:
            raise ValueError("Invalid particle")
        if displacement.shape!=self.positions[0].shape:
            raise ValueError("Wrong displacement dimension")
        new_particle_position = (self.positions[k] + displacement) % self.box_size
        rel_pos = self.positions[[i for i in range(self.n_particles) if i!=k]] - new_particle_position
        rel_pos -= self.box_size * np.rint(rel_pos/self.box_size)
        dis = np.linalg.norm(rel_pos,axis=1)
        #reject
        if (dis<2*self.particle_radius).any():
            return False #to say it was rejected
        
        self.positions[k] = new_particle_position
        return True #indication that was accepted