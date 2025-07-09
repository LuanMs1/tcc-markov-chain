import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sigma = 0.1
dt = 0.05
T = 2
n_steps = int(T / dt)

positions = np.array([[0.2, 0.5], [0.8, 0.5], [0.5, 0.8], [0,0], [0,.3], [.5,.2]])
velocities = 20 * np.array([[0.3, 0.3], [-0.6, 0], [-0.2, -0.3],[0,0],[0,0],[0,.2]])
N = len(positions)
history = np.zeros((n_steps, N, 2))
history[0] = positions

def compute_next_collision(positions, velocities, sigma):
    soonest_time = np.inf
    pair = None
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i] - positions[j]
            dv = velocities[i] - velocities[j]
            dx_dv = np.dot(dx, dv)
            if dx_dv >= 0:
                continue
            dv_dv = np.dot(dv, dv)
            dx_dx = np.dot(dx, dx)
            delta = dx_dv**2 - dv_dv * (dx_dx - 4 * sigma**2)
            if delta < 0:
                continue
            sqrt_delta = np.sqrt(delta)
            t1 = (-dx_dv + sqrt_delta) / dv_dv
            t2 = (-dx_dv - sqrt_delta) / dv_dv
            t = min(t1, t2)
            if 0 < t < soonest_time:
                soonest_time = t
                pair = (i, j)
    return soonest_time, pair

# Run simulation
current_time = 0.0
step = 1
while current_time < T and step < n_steps:
    t_col, pair = compute_next_collision(positions, velocities, sigma)
    t_advance = min(dt, t_col)
    positions += velocities * t_advance
    current_time += t_advance
    if t_col <= dt and pair:
        i, j = pair
        dx = positions[i] - positions[j]
        dv = velocities[i] - velocities[j]
        dx_norm_sq = np.dot(dx, dx)
        factor = np.dot(dv, dx) / dx_norm_sq
        velocities[i] -= factor * dx
        velocities[j] += factor * dx
    history[step] = positions
    step += 1

# Animate
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
circles = [plt.Circle((0, 0), sigma, fc='blue') for _ in range(N)]
for c in circles:
    ax.add_patch(c)

def animate(frame):
    for i, c in enumerate(circles):
        c.center = history[frame, i]
    return circles

ani = FuncAnimation(fig, animate, frames=step, interval=50, blit=True)
ani.save("simulation.gif", fps=30)