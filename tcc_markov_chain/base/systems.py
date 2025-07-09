from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class HardDiskSystem():
    box_size:float = 1
    particle_radius:float = 0.1
    n_particles:int = 5
    seed: int = 42
    max_velocity:int = 1
    periodict_boundary:bool = False
    rng: np.random.Generator = field(init=False)
    positions:np.ndarray = field(init=False)
    velocities:np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.positions = self._set_positions()
        self.velocities = self._set_velocities()

    def _set_positions(self):
        '''set the positions of the particles'''
        positions = []
        while len(positions) < self.n_particles:
            margin = self.periodict_boundary * self.particle_radius
            pos = self.rng.uniform(0 + margin, self.box_size - margin, size = 2)
            if all(np.linalg.norm(pos - p) > 2*self.particle_radius for p in positions):
                positions.append(pos)
        return np.array(positions)
    
    def _set_velocities(self):
        return np.array([self.rng.uniform(-self.max_velocity,self.max_velocity,size=2) for i in range(self.n_particles)])
    
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