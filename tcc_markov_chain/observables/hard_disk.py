from tcc_markov_chain.base_simulation import Simulation
from tcc_markov_chain.systems import DiskSystem
from typing import List
import numpy as np

#only work for square boxes
def particle_density(system:DiskSystem, strip_count = 10)->List[float]:
    strip_size = system.box_dimension[0] / strip_count
    strips = np.arange(strip_count) * strip_size
    strip_contribution = []
    for strip in strips:
        is_inside = (system.positions[:,1] >= strip) * (system.positions[:,1] < strip + strip_size)
        factor = np.sum(is_inside)
        strip_contribution.append(factor)
    # system.strip_density += strip_contribution
    return strip_contribution

#only work for square boxes
def rdf(system:DiskSystem, n_bins=10):
    distances = system.calculate_relative_positions()
    distances = np.linalg.norm(distances,axis=1)
    bin_size = system.box_dimension[0] / 2*n_bins
    bins =np.linspace(0, system.box_dimension[0]/2, n_bins + 1)
    for d in distances:
        bins[int(d//bin_size)]+=2
    return bins