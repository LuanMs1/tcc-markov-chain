from .systems import HardDiskSystem,NHDWalls,Event
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from typing import Tuple, Callable, Any, Optional, Dict
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)

class HardDiskSimulation(ABC):
    def __init__(self, system:HardDiskSystem, dt:float=0.5, event_driven:bool=False):
        self.system= system
        self.time=0
        self.dt=dt
        self.event_driven=event_driven
        self.collision_triggered = True
    
    def set_time_interval(self,dt:float):
        self.dt=dt
    
    @abstractmethod
    def step(self):
        """advance the system one step"""
    
    @abstractmethod
    def run(self, n_steps:int):
        """run the simulation"""

    def reset(self):
        """resets """
        self.system.positions = self.system.initial_positions
        self.system.velocities = self.system.initial_velocities
        self.time = 0
    
class BoundarySimulation(HardDiskSimulation):
    def __init__(self, system:NHDWalls, debug = False,dt=0.5,event_driven=False):
        super().__init__(system,dt,event_driven=event_driven)
        self.debug = debug
        log.info(f'Simulation enviorment with walls created.')

    def step(self):
        pass

    def run(
            self,
            n_steps:float = 5.,
            fn:Callable = None,
        ):
        final_t = self.time + (self.dt * n_steps)
        next_frame = self.time + self.dt
        log.info(f"running simulation from {self.time} to {final_t} in {self.dt}")
        next_event = self.system._get_next_event(time=self.time)
        pbar = tqdm(total=n_steps, desc="Simulating", unit="frame")
        frame_count=0
        while self.time < final_t:
            if (self.time < 0)|(next_event.time<0):
                log.warning("time of the system or of the event is negative")
                break
            if next_event.time < next_frame:
                self.system._update_positions(next_event.time - self.time)
                self.time=next_event.time
                next_event.call_trigger()
                next_event = self.system._get_next_event(time=self.time)
                continue
            
            self.system._update_positions(next_frame-self.time)
            #evaluate
            fn(self)

            self.time=next_frame
            frame_count += 1
            pbar.update(1)

            next_frame+=self.dt
        pbar.close()