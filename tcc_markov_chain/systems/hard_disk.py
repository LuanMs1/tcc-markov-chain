from .base import DiskSystem
from dataclasses import dataclass
import numpy as np


from typing import Tuple, Callable, Any, Optional, Dict

@dataclass
class HDPeriodic(DiskSystem):

    def calculate_relative_positions(self, positions:Optional[np.ndarray] = None):
        if not isinstance(positions,np.ndarray):
            positions=self.positions.copy()
        # else:
        #     new_positions = self.positions.copy()
        # self._basic_positions_check(new_positions)
    
        i_idx, j_idx = np.triu_indices(positions.shape[0],k=1)
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
        
        new_pos = self.positions.copy()
        new_pos[k] = (new_pos[k] + displacement) % self.box_size
        return new_pos