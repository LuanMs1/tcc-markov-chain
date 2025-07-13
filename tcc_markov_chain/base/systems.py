from dataclasses import dataclass, field
from typing import Tuple
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)

## Improve logging
## Add animation

@dataclass
class HardDiskSystem():
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    seed: int = 42
    max_velocity:int = 1
    periodic_boundary:bool = False
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)

    def _error_check(self):
        if len(self.positions.shape) != 2:
            raise ValueError("array of positions needs to have 2 dimensions")
        if len(self.velocities.shape) != 2:
            raise ValueError("array of velocities needs to have 2 dimensions")
        if 2*self.particle_radius > self.box_size:
            raise ValueError("particle radius dyameter needs to be less them box_size")
        
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_positions()
        self.velocities = self._set_velocities()
        self._error_check()
        log.info(f'system created with {self.n_particles} particles')


    def _set_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = (not self.periodic_boundary) * self.particle_radius
            pos = self.rng.uniform(0 + margin, self.box_size - margin, size = 2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def _set_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
    def get_relative_positions(self) -> np.ndarray:
        """
        get the distances between particle pairs 
        Returns:
            distances (np.ndarray): pairwise distances.
        """
        # particle_pairs (List[Tuple[int, int]]): List of (i, j) index pairs corresponding to each distance.

        relative_positions = self.positions[:,np.newaxis,:] - self.positions[np.newaxis,:,:]
        return relative_positions
    
    def get_relative_velocities(self) -> np.ndarray:
        relative_velocities = self.velocities[:,np.newaxis,:] - self.velocities[np.newaxis,:,:]
        return relative_velocities
    
    def plot_system(self, show_velocities=True, ax = None, show_grid = False):
        if ax == None:
            fig, ax = plt.subplots()    
        
        # Plotando os discos
        for pos in self.positions:
            circle = plt.Circle(pos, self.particle_radius, color='blue', alpha=0.5)
            ax.add_patch(circle)
        
        # Plotando as velocidades como setas
        if show_velocities:
            ax.quiver(
                self.positions[:, 0], self.positions[:, 1],  # x, y das partículas
                self.velocities[:, 0], self.velocities[:, 1],  # componentes vx, vy
                color='black', angles='xy', scale_units='xy', scale=1, width=0.005
            )
        
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_aspect('equal')
        ax.set_title("Hard Disk System")
        plt.grid(show_grid)
        # plt.show()

class BoundarySimulation():

    def __init__(self, system:HardDiskSystem, debug = False):
        if system.periodic_boundary:
            error_message = "the system must have aperiodic boundary"
            log.error(error_message)
            raise ValueError(error_message)
        self.system = system
        self.time = 0.
        self.debug = debug
        log.info(f'Simulation enviorment with aperiodic boundary created.')

    # def particle_wall_colision(sel)
    def _next_wall_colision(self) -> Tuple[np.array, np.array]:
        """
            calculate the wall colisions for each particle.
            return the first wall that each particle will colide
        """
        pos_mask = self.system.velocities > 0
        neg_mask = self.system.velocities < 0

        #set the direction mask
        t_plus = np.full_like(self.system.positions, np.inf)
        t_neg = np.full_like(self.system.positions, np.inf)

        #calculate the time of colision in each direction
        pos_dis = self.system.box_size - self.system.particle_radius - self.system.positions[pos_mask]
        t_plus[pos_mask] = pos_dis / self.system.velocities[pos_mask]
        neg_dis = self.system.positions[neg_mask] - self.system.particle_radius
        t_neg[neg_mask] = neg_dis / -self.system.velocities[neg_mask]

        #set the minimun colision time in each dimension for the particles
        times = np.minimum(t_neg, t_plus)
        #wall that was hitted (0 for colision in y and 1 for colision in x)
        wall_hit = np.argmin(times,axis=1)
        #smallest interval for next colision for each particle
        hit_times = times[np.arange(self.system.n_particles), wall_hit]
        
        colided_particle = np.argmin(hit_times)
        wall_colided = wall_hit[colided_particle]
        time_to_colide = min(hit_times)
        return wall_colided, colided_particle, time_to_colide
    
    def _wall_colision(self, particle_index, wall_axis):
        self.system.velocities[particle_index][wall_axis] *= -1

    def _next_pair_colision(self):
        rel_dis = self.system.get_relative_positions()
        rel_vel = self.system.get_relative_velocities()

        dx_dv = np.einsum("ijk,ijk->ij",rel_dis,rel_vel)
        dx_dx = np.einsum("ijk,ijk->ij",rel_dis,rel_dis)
        dv_dv = np.einsum("ijk,ijk->ij",rel_vel,rel_vel)

        delta = np.square(dx_dv) - dv_dv * (dx_dx - 4*self.system.particle_radius**2)
        mask = (delta>-1e-12) & (dx_dv<0)
        np.fill_diagonal(mask, False)

        dt = np.full_like(delta, np.inf, dtype=float)
        dt[mask] = - (dx_dv[mask] + np.sqrt(delta[mask])) / dv_dv[mask]

        i,j = np.where(dt == dt.min())

        return (i[0], j[0]), dt[i[0], j[0]]

    def _pair_colision(self, i, j):
        '''update particle i,j velocities'''
        #calcular vetor normal ao plano de colisao
        dx = self.system.positions[i] - self.system.positions[j]
        dv = self.system.velocities[i] - self.system.velocities[j]

        colision_normal = dx / np.sqrt(np.dot(dx,dx))
        magnitude_change = np.dot(colision_normal,dv)
        
        self.system.velocities[i] = self.system.velocities[i] - magnitude_change * colision_normal
        self.system.velocities[j] = self.system.velocities[j] + magnitude_change * colision_normal

    def _update_position(self, dt):
        self.system.positions = self.system.positions + (self.system.velocities * dt)
        self.time += dt
        
    def step(self):
        
        if self.debug:
            log.info("positions: ")
            log.info(self.system.positions)
            log.info("velocities: ")
            log.info(self.system.velocities)


        axis_index, wall_particle, wall_dt = self._next_wall_colision()
        (pi, pj), pair_dt = self._next_pair_colision()

        if self.debug:
            next_colision_type = "wall" if wall_dt < pair_dt else "pair"
            next_colision_time = self.time + min(wall_dt, pair_dt)
            log.info(f"next colision is{next_colision_type} at {next_colision_time}")
            log.info(f"colision will happen after {min(wall_dt, pair_dt)}")

        self._update_position(min(wall_dt, pair_dt))
        if wall_dt < pair_dt:
            self._wall_colision(wall_particle, axis_index)
        elif wall_dt == pair_dt:
            self._wall_colision(wall_particle, axis_index)
            self._pair_colision(pi, pj)
        else:
            self._pair_colision(pi, pj)

        if self.debug:
            log.info(f"new positions: {self.system.positions}" )
            log.info(f"new velocities: {self.system.velocities}")
            log.info(f"new tiem: {self.time}")

    def run(self, n_steps = 5, plot_each_step = False, ax = None, debug = False):
        '''simulate all steps'''
        log.info(f'Running simulation with {n_steps} events')
        self.debug = debug
        for i in range(n_steps):
            if isinstance(ax, np.ndarray):
                step_ax = ax[i]
            else:
                step_ax = ax
            if self.debug:
                log.info(f"step {i} of simulation")
            if plot_each_step | (i==0):
                self.system.plot_system(ax = step_ax)
            self.step()
        
        self.system.plot_system(ax=ax[-1] if isinstance(ax,np.ndarray) else ax)
        log.info('end of simulation')

