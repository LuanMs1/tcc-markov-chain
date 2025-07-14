from .systems import HardDiskSystem#,NHDWalls,Event
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from typing import Tuple, Callable, Any, Optional, Dict
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)

class BaseSimulation(ABC):
    def __init__(self, system:HardDiskSystem):
        self.sys = system

    @abstractmethod
    def step(self):
        """run each step of simulation"""

    def run(self, eval_fn:Callable, n_steps:int=5):
        """run the simulation"""
        pbar = tqdm(total=n_steps, desc="Simulating", unit="frame")
        frame_count=0
        for i in range(n_steps):
            self.step()
            eval_fn(self)
            frame_count += 1
            pbar.update(1)
        pbar.close()

class DirectSampling(BaseSimulation):
    def __init__(self,system:HardDiskSystem):
        super().__init__(system=system)

    def direct_sampling(self):
        margin = self.sys.particle_radius

        new_particle = self.sys.rng.uniform(0+margin,self.sys.box_size-margin,size=2)
        new_system_positions = np.array([new_particle])
        while new_system_positions.shape[0] < self.sys.n_particles:
            
            for i in range(self.sys.n_particles-1):
                new_particle = self.sys.rng.uniform(0+margin,self.sys.box_size-margin,size=2)
                dis = np.linalg.norm(new_system_positions-new_particle,axis=1)
                if (dis <= 2*self.sys.particle_radius).any():
                    new_system_positions = np.array([new_particle])
                    break
                new_system_positions = np.vstack((new_system_positions,new_particle.reshape(1,-1)))
        return new_system_positions
    def step(self):
        self.sys.positions = self.direct_sampling()   

               

