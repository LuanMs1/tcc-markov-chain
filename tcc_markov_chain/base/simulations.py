from .systems import HardDiskSystem
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Callable, Any, Optional, Dict
import utils.logging as logger
logger.setup_logging()
log = logger.get_logger(__name__)

from dataclasses import dataclass
@dataclass
class Event:
    time:float
    trigger:Callable[...,None]
    func_args:Dict[str,Any]

    def call_trigger(self):
        self.trigger(**self.func_args)

class HardDiskSimulation(ABC):
    def __init__(self, system:HardDiskSystem, dt:float=0.5, event_driven:bool=False):
        self.system= system
        self.time=0
        self.dt=dt
        self.event_driven=event_driven
        self.collision_triggered = True
    
    def set_time_interval(self,dt:float):
        self.dt=dt

    def set_simulation_type(self,event_driven:bool):
        """
            set the type of simulation:
            event_driven=True: each step jump to the next event
            event_driven=False: each step based on dt
        """
        self.event_driven=event_driven

    def _next_pair_collision(self):
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

    def _pair_collision(self, i, j):
        '''update particle i,j velocities'''
        #calcular vetor normal ao plano de colisao
        dx = self.system.positions[i] - self.system.positions[j]
        dv = self.system.velocities[i] - self.system.velocities[j]

        collision_normal = dx / np.sqrt(np.dot(dx,dx))
        magnitude_change = np.dot(collision_normal,dv)
        
        self.system.velocities[i] = self.system.velocities[i] - magnitude_change * collision_normal
        self.system.velocities[j] = self.system.velocities[j] + magnitude_change * collision_normal

    def _update_position(self, dt):
        self.system.positions = self.system.positions + (self.system.velocities * dt)
        self.time += dt
    
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
    def __init__(self, system:HardDiskSystem, debug = False,dt=0.5,event_driven=False):
        super().__init__(system,dt,event_driven=event_driven)
        if self.system.periodic_boundary:
            error_message = "the system must have aperiodic boundary"
            log.error(error_message)
            raise ValueError(error_message)
        self.debug = debug
        log.info(f'Simulation enviorment with aperiodic boundary created.')

    # def particle_wall_collision(sel)
    def _next_wall_collision(self) -> Tuple[np.array, np.array]:
        """
            calculate the wall collisions for each particle.
            return the first wall that each particle will colide
        """
        pos_mask = self.system.velocities > 0
        neg_mask = self.system.velocities < 0

        #set the direction mask
        t_plus = np.full_like(self.system.positions, np.inf)
        t_neg = np.full_like(self.system.positions, np.inf)

        #calculate the time of collision in each direction
        pos_dis = self.system.box_size - self.system.particle_radius - self.system.positions[pos_mask]
        t_plus[pos_mask] = pos_dis / self.system.velocities[pos_mask]
        neg_dis = self.system.positions[neg_mask] - self.system.particle_radius
        t_neg[neg_mask] = neg_dis / -self.system.velocities[neg_mask]

        #set the minimun collision time in each dimension for the particles
        times = np.minimum(t_neg, t_plus)
        #wall that was hitted (0 for collision in y and 1 for collision in x)
        wall_hit = np.argmin(times,axis=1)
        #smallest interval for next collision for each particle
        hit_times = times[np.arange(self.system.n_particles), wall_hit]
        
        colided_particle = np.argmin(hit_times)
        wall_colided = wall_hit[colided_particle]
        time_to_colide = min(hit_times)
        return wall_colided, colided_particle, time_to_colide
    
    def _wall_collision(self, particle_index, wall_axis):
        self.system.velocities[particle_index][wall_axis] *= -1

    def _simultaneous_collision(self, pi,pj, wall_particle, wall_axis):
        self._pair_collision(pi,pj)
        self._wall_collision(wall_particle,wall_axis)

    def _get_next_event(self) -> Event:
        axis_index, wall_particle, wall_dt = self._next_wall_collision()
        (pi, pj), pair_dt = self._next_pair_collision()

        if wall_dt==pair_dt:
            func_arguments = {
                'pi':pi,
                'pj':pj,
                'wall_particle':wall_particle,
                'wall_axis':axis_index
            }
            return Event(wall_dt+self.time,self._simultaneous_collision,func_arguments)
        if wall_dt<pair_dt:
            func_arguments = {
                'particle_index':wall_particle,
                'wall_axis':axis_index
            }
            return Event(wall_dt+self.time,self._wall_collision, func_arguments)
        else:
            func_arguments = {
                'i':pi,
                'j':pj
            }
            return Event(pair_dt+self.time,self._pair_collision, func_arguments)


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
        next_event = self._get_next_event()
        while self.time < final_t:
            
            if next_event.time < next_frame:
                self._update_position(next_event.time - self.time)
                self.time=next_event.time
                next_event.call_trigger()
                next_event = self._get_next_event()
                continue
            
            self._update_position(next_frame-self.time)
            #evaluate
            fn(self)

            self.time=next_frame
            next_frame+=self.dt