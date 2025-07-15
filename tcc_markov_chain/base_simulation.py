from abc import ABC, abstractmethod
from .systems import DiskSystem
from typing import Any
from tqdm import tqdm

class Simulation(ABC):
    def __init__(self, system:DiskSystem):
        self.system = system

    def run(
        self, 
        eval_fn:callable, 
        n_steps:int=5,
        #method?
        #safe run? Some way to stop infinity runs
        #stop rule? stop by a condition?
    )->Any:
        """run the simulation"""
        # pbar = tqdm(total=n_steps, desc="Simulating", unit="frame")
        # frame_count=0
        for i in tqdm(range(n_steps)):
            self.step()
            yield eval_fn(self)
            # frame_count += 1
            # pbar.update()
        # pbar.close()

    @abstractmethod
    def step(self):
        """run each step"""
        pass
