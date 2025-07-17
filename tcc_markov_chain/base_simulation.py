from abc import ABC, abstractmethod
from .systems import DiskSystem
from typing import Any
from tqdm import tqdm

class Simulation(ABC):
    def __init__(self, system:DiskSystem):
        self.system = system

    def run(
        self, 
        eval_fn:Any, 
        n_steps:int=5,
        #method?
        #safe run? Some way to stop infinity runs
        #stop rule? stop by a condition?
    )->Any:
        """run the simulation"""
        
        for i in tqdm(range(n_steps)):
            self.step()
            yield eval_fn(self.system)

    @abstractmethod
    def step(self):
        """run each step"""
        pass
