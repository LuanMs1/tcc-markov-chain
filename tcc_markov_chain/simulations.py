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

        _ = self.system.update_particle_position(k,dx)
        # valid_move = self.system.validate_configuration(new_system_positions)

    def step(self):
        return self.chain_move(delta_x=self.delta_x)

class PeriodicDirectSampling(Simulation):
    def __init__(self,system):
        super().__init__(system=system)

    def direct_sampling(self):
        for i in range(self.system.n_particles):
            x = self.system.rng.uniform(0,self.system.box_dimension[0])
            y = self.system.rng.uniform(0,self.system.box_dimension[1])
            new_particle = np.array([x,y])
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

# class PeriodicCluster(Simulation):
#     #dont know. Probably isn't necessary
#     def __init__(self,system, symmetry_axis = None):
#         super().__init__(system=system)
#         if symmetry_axis is None:
#             self.symmetry_axis = self.system.box_dimension[0] / 2
#         self.symmetry_axis = symmetry_axis

#     def cluster_sampling(self):
#         x_move = 2*(self.symmetry_axis - self.system.positions[:,0])
#         for i in range(self.system.n_particles):
#             particle = self.system.positions[i].copy()
#             # move particle to the other side of the symmetry axis
#             particle[0] = (particle[0] + x_move) % self.system.box_dimension[0]
#             rel_pos = particle - self.system.positions[[k for k in range(self.system.n_particles) if k != i]]
#             rel_pos -= self.system.box_dimension * np.rint(rel_pos / self.system.box_dimension)
#             distances = np.linalg.norm(rel_pos, axis=1)
#             # if a particle is in supperposition, move it to the other side

#             #check if it colides with any other particle
#             rel_pos = self.system.single_particle_relative_positions(new_position)
#         # mirror_positions = self.system.positions.copy()
#         # mirror_positions[:,0] = (
#         #     mirror_positions[:,0] + x_move
#         # ) % self.system.box_dimension[0]
#         # old_and_mirror = self.system.positions.copy()
#         # old_and_mirror = np.vstack((old_and_mirror, mirror_positions))
#         # rel_dis = self.system.calculate_relative_positions(old_and_mirror)
#         # dis = np.linalg.norm(rel_dis, axis=1)
#         # superpos
#         # self.system.positions = old_and_mirror#[dis.any(axis=1)]
#         # self.system.plot_system(labels=True)
    
#     def step(self):
#         sampling = True
#         while sampling:
#             sampling = self.cluster_sampling()