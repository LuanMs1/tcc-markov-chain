import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configurações
sigma = 0.05
dt = 0.01
T = 4
n_steps = int(T / dt)
box_size = 1.0

# Inicialização
# positions = np.array([
#     [0.10, 0.20], [0.90, 0.80], [0.30, 0.70], [0.70, 0.25], [0.45, 0.90],
#     [0.15, 0.85], [0.80, 0.35], [0.55, 0.10], [0.25, 0.50], [0.65, 0.65],
#     [0.40, 0.30], [0.60, 0.90], [0.35, 0.15], [0.85, 0.15], [0.20, 0.75],
#     [0.75, 0.50], [0.50, 0.45], [0.60, 0.20], [0.90, 0.40], [0.10, 0.60]
# ])
# velocities = 10 * np.array([
#     [ 0.10,  0.05], [-0.08,  0.07], [ 0.04, -0.10], [-0.06,  0.09], [ 0.05,  0.03],
#     [ 0.07, -0.06], [-0.04,  0.02], [ 0.06, -0.08], [-0.07,  0.04], [ 0.02, -0.03],
#     [ 0.09,  0.01], [-0.05,  0.06], [ 0.03,  0.08], [ 0.01, -0.07], [-0.02,  0.09],
#     [-0.03, -0.05], [ 0.08,  0.02], [-0.06, -0.06], [ 0.04,  0.07], [-0.09, -0.02]
# ])


positions = np.array([[0.2, 0.5], [0.8, 0.5], [0.5, 0.8]])
velocities = 2 * np.array([[0.4, 0.3], [-0.6, 0], [-0.2, -0.3]])
N = len(positions)
history = np.zeros((n_steps, N, 2))
history[0] = positions

def compute_next_disk_collision(pos, vel, sigma):
    soonest_time = np.inf
    pair = None
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i] - pos[j]
            dv = vel[i] - vel[j]
            dx_dv = np.dot(dx, dv)
            if dx_dv >= 0: continue
            dv_dv = np.dot(dv, dv)
            dx_dx = np.dot(dx, dx)
            delta = dx_dv**2 - dv_dv * (dx_dx - 4 * sigma**2)
            if delta < 0: continue
            sqrt_delta = np.sqrt(delta)
            t = min((-dx_dv + sqrt_delta) / dv_dv, (-dx_dv - sqrt_delta) / dv_dv)
            if 0 < t < soonest_time:
                soonest_time = t
                pair = (i, j)
    return soonest_time, pair

def compute_wall_collision_time(pos, vel, sigma):
    t_wall = np.full(N, np.inf)
    for i in range(N):
        times = []
        for d in range(2):
            if vel[i, d] > 0:
                t = (box_size - sigma - pos[i, d]) / vel[i, d]
            elif vel[i, d] < 0:
                t = (sigma - pos[i, d]) / vel[i, d]
            else:
                t = np.inf
            times.append(t)
        t_wall[i] = min(times)
    return t_wall

# Simulação
current_time = 0.0
step = 1
while current_time < T and step < n_steps:
    t_col, pair = compute_next_disk_collision(positions, velocities, sigma)
    t_walls = compute_wall_collision_time(positions, velocities, sigma)
    t_wall_min = np.min(t_walls)
    i_wall = np.argmin(t_walls)

    t_next = min(dt, t_col, t_wall_min)
    positions += velocities * t_next
    current_time += t_next

    if t_col <= t_next and pair:
        i, j = pair
        dx = positions[i] - positions[j]
        dv = velocities[i] - velocities[j]
        dx_norm_sq = np.dot(dx, dx)
        factor = np.dot(dv, dx) / dx_norm_sq
        velocities[i] -= factor * dx
        velocities[j] += factor * dx

    if t_wall_min <= t_next:
        for d in range(2):
            if velocities[i_wall, d] > 0 and positions[i_wall, d] + sigma >= box_size:
                velocities[i_wall, d] *= -1
            elif velocities[i_wall, d] < 0 and positions[i_wall, d] - sigma <= 0:
                velocities[i_wall, d] *= -1

    history[step] = positions
    step += 1

# Animação
fig, ax = plt.subplots()
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
circles = [plt.Circle((0, 0), sigma, fc='blue') for _ in range(N)]
for circle in circles:
    ax.add_patch(circle)

def animate(frame):
    for i, circle in enumerate(circles):
        circle.center = history[frame, i]
    return circles

ani = FuncAnimation(fig, animate, frames=step, interval=100, blit=True)
ani.save("simulation_walls.gif", fps=30)
print("GIF salvo como 'simulation.gif'")
