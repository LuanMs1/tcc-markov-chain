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
            yield eval_fn(self)
            frame_count += 1
            pbar.update()
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
                new_system_positions = np.vstack((new_system_positions,new_particle.reshape(1,-1)))
                is_valid = self.sys._check_particles_superposition(new_system_positions, i=i+1)
                if not is_valid:
                    new_system_positions = np.array([new_particle])
                    break
        return new_system_positions
    def step(self):
        self.sys.positions = self.direct_sampling()   

class MarkovChain(BaseSimulation):
    def __init__(self, system:HardDiskSystem,delta_x:float):
        """prepare simulation with steps of delta_x"""
        super().__init__(system=system)
        self.delta_x=delta_x

    def chain_move(self, delta_x:float):
        "one move in random direction of max displacement delta_x"
        k = self.sys.rng.integers(0,self.sys.n_particles)
        dx = self.sys.rng.uniform(-delta_x,delta_x,size=2)

        new_system_positions = self.sys._update_particle_position(k,dx)
        valid_move = self.sys._check_particles_superposition(positions=new_system_positions,i=k)

        if valid_move:
            self.sys.positions = new_system_positions

    def step(self):
        self.chain_move(delta_x=self.delta_x)
