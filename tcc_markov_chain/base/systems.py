from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, Any
from abc import ABC,abstractmethod
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)

############ DISK SYSTEMS FOR NEWTONIAN SIMULATIONS ###############
## Improve logging
## Add animation

@dataclass
class Event:
    time:float
    trigger:Callable[...,None]
    func_args:Dict[str,Any]

    def call_trigger(self):
        self.trigger(**self.func_args)

@dataclass
class HardDiskSystem(ABC):
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    seed: int = 42
    max_velocity:int = 1
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)

    def _error_check(self):
        # if len(self.positions.shape) != 2:
        #     raise ValueError("array of positions needs to have 2 dimensions")
        # if len(self.velocities.shape) != 2:
        #     raise ValueError("array of velocities needs to have 2 dimensions")
        if 2*self.particle_radius > self.box_size:
            raise ValueError("particle radius dyameter needs to be less them box_size")
        
    def __post_init__(self):
        self._error_check()
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_random_positions()
        self.velocities = self._set_random_velocities()
        self.initial_positions = self.positions.copy()
        self.initial_velocities = self.velocities.copy()
        log.info(f'system created with {self.n_particles} particles')

    def _set_random_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = self.particle_radius
            pos = self.rng.uniform(0 + margin, self.box_size - margin, size = 2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def _set_random_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
    def set_particle_parameters(self,positions:np.ndarray=np.array([]),velocities:np.ndarray=np.array([])):
        
        pos_n = positions.shape[0]
        vel_n = velocities.shape[0]
        
        if (pos_n+vel_n) == 0:
            log.warning('no change')
            return
        
        if (pos_n+vel_n)>pos_n:
            if pos_n != vel_n:
                raise ValueError("positions and velocities needs to have same size")
            if pos_n != self.n_particles:
                raise ValueError("You cannot change the system syze, please build a new one")
        
        if pos_n>0:
            self.positions=positions
        if vel_n>0:
            self.velocities=velocities

    def _pair_collision(self, i, j):
        '''update particle i,j velocities'''
        #calcular vetor normal ao plano de colisao
        dx = self.positions[i] - self.positions[j]
        dv = self.velocities[i] - self.velocities[j]

        collision_normal = dx / np.sqrt(np.dot(dx,dx))
        magnitude_change = np.dot(collision_normal,dv)
        
        self.velocities[i] = self.velocities[i] - magnitude_change * collision_normal
        self.velocities[j] = self.velocities[j] + magnitude_change * collision_normal

            
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
    
    @abstractmethod
    def _update_positions(self):
        """update posistion of particles"""
        pass
    @abstractmethod
    def get_relative_positions(self) ->np.ndarray:
        """
        get the distances between particle pairs 
        Returns:
            distances (np.ndarray): pairwise relative positions.
        """
        pass

    @abstractmethod
    def get_relative_velocities(self) -> np.ndarray:
        pass


@dataclass
class NHDWalls(HardDiskSystem):

    def get_relative_positions(self) -> np.ndarray:
        # particle_pairs (List[Tuple[int, int]]): List of (i, j) index pairs corresponding to each distance.
        relative_positions = self.positions[:,np.newaxis,:] - self.positions[np.newaxis,:,:]
        return relative_positions
    
    def get_relative_velocities(self) -> np.ndarray:
        relative_velocities = self.velocities[:,np.newaxis,:] - self.velocities[np.newaxis,:,:]
        return relative_velocities
    
    def _update_positions(self, dt):
        self.positions = self.positions + (self.velocities * dt)

    def _next_pair_collision(self):
        rel_dis = self.get_relative_positions()
        rel_vel = self.get_relative_velocities()

        dx_dv = np.einsum("ijk,ijk->ij",rel_dis,rel_vel)
        dx_dx = np.einsum("ijk,ijk->ij",rel_dis,rel_dis)
        dv_dv = np.einsum("ijk,ijk->ij",rel_vel,rel_vel)

        delta = np.square(dx_dv) - dv_dv * (dx_dx - 4*self.particle_radius**2)
        mask = (delta>0) & (dx_dv<0)
        np.fill_diagonal(mask, False)

        dt = np.full_like(delta, np.inf, dtype=float)
        dt[mask] = - (dx_dv[mask] + np.sqrt(delta[mask])) / dv_dv[mask]
        i,j = np.where(dt == dt.min())        

        return (i[0], j[0]), dt[i[0], j[0]]

    # def particle_wall_collision(sel)
    def _next_wall_collision(self) -> Tuple[np.array, np.array]:
        """
            calculate the wall collisions for each particle.
            return the first wall that each particle will colide
        """
        pos_mask = self.velocities > 0
        neg_mask = self.velocities < 0

        #set the direction mask
        t_plus = np.full_like(self.positions, np.inf)
        t_neg = np.full_like(self.positions, np.inf)

        #calculate the time of collision in each direction
        pos_dis = self.box_size - self.particle_radius - self.positions[pos_mask]
        t_plus[pos_mask] = pos_dis / self.velocities[pos_mask]
        neg_dis = self.positions[neg_mask] - self.particle_radius
        t_neg[neg_mask] = neg_dis / -self.velocities[neg_mask]

        #set the minimun collision time in each dimension for the particles
        times = np.minimum(t_neg, t_plus)
        #wall that was hitted (0 for collision in y and 1 for collision in x)
        wall_hit = np.argmin(times,axis=1)
        #smallest interval for next collision for each particle
        hit_times = times[np.arange(self.n_particles), wall_hit]
        
        colided_particle = np.argmin(hit_times)
        wall_colided = wall_hit[colided_particle]
        time_to_colide = min(hit_times)
        return wall_colided, colided_particle, time_to_colide
    
    def _wall_collision(self, particle_index, wall_axis):
        self.velocities[particle_index][wall_axis] *= -1

    def _simultaneous_collision(self, pi,pj, wall_particle, wall_axis):
        self._pair_collision(pi,pj)
        self._wall_collision(wall_particle,wall_axis)

    def _get_next_event(self,time=0) -> Event:
        axis_index, wall_particle, wall_dt = self._next_wall_collision()
        (pi, pj), pair_dt = self._next_pair_collision()

        if wall_dt==pair_dt:
            func_arguments = {
                'pi':pi,
                'pj':pj,
                'wall_particle':wall_particle,
                'wall_axis':axis_index
            }
            return Event(wall_dt+time,self._simultaneous_collision,func_arguments)
        if wall_dt<pair_dt:
            func_arguments = {
                'particle_index':wall_particle,
                'wall_axis':axis_index
            }
            return Event(wall_dt+time,self._wall_collision, func_arguments)
        else:
            func_arguments = {
                'i':pi,
                'j':pj
            }
            return Event(pair_dt+time,self._pair_collision, func_arguments)
