from .base_simulation import Simulation
import numpy as np

class MarkovChain(Simulation):
    def __init__(self, system,delta_x:float):
        super().__init__(system=system)
        self.delta_x = delta_x

    def chain_move(self, delta_x:float):
        "one move in random direction of max displacement delta_x"
        k = self.system.rng.integers(0,self.system.n_particles)
        dx = self.system.rng.uniform(-delta_x,delta_x,size=2)

        new_system_positions = self.system.update_particle_position(k,dx)
        valid_move = self.system.validate_configuration(new_system_positions)

        if valid_move:
            self.system.positions = new_system_positions

    def step(self):
        self.chain_move(delta_x=self.delta_x)

class PeriodicDirectSampling(Simulation):
    def __init__(self,system):
        super().__init__(system=system)

    def direct_sampling(self):
        for i in range(self.system.n_particles):
            new_particle = self.system.rng.uniform(0,self.system.box_size,size=2)
            if i==0:
                new_system_positions = np.array([new_particle],dtype=float)
            else:
                new_system_positions = np.vstack((new_system_positions,new_particle.reshape(1,-1)))
            
            if self.system.validate_configuration(new_system_positions):
                continue
            else:
                return True
        self.system.set_positions(new_system_positions)
        return False
    def step(self):
        sampling = True
        while sampling:
            sampling = self.direct_sampling()


