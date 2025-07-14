
import matplotlib.animation as animation

def animate_simulation(
    sim: BoundarySimulation,
    n_frames: int,
    dt_frame: float,
    interval:  int = 50,
    figsize:   tuple[int,int] = (6,6)
) -> animation.FuncAnimation:
    """
    Create and return a FuncAnimation for `sim`.
    
    Parameters
    ----------
    sim
        A BoundarySimulation (or subclass) instance, already initialized.
    n_frames
        How many frames to draw.
    dt_frame
        The simulation-time per frame; must match sim.dt or set sim.dt=dt_frame.
    interval
        Milliseconds between frames in the GUI.
    figsize
        Size of the matplotlib figure.
    """
    # ensure sim.dt matches
    sim.set_time_interval(dt_frame)

    # build the figure & artists
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, sim.system.box_size)
    ax.set_ylim(0, sim.system.box_size)
    ax.set_aspect('equal')
    ax.set_title("Hard Disk Event‐Driven MD")

    # initial draw of disks
    circles = []
    for p in sim.system.positions:
        c = plt.Circle(p, sim.system.particle_radius, color='blue', alpha=0.5)
        ax.add_patch(c)
        circles.append(c)

    # single quiver for all velocities
    quiv = ax.quiver(
        sim.system.positions[:,0],
        sim.system.positions[:,1],
        sim.system.velocities[:,0],
        sim.system.velocities[:,1],
        angles='xy', scale_units='xy', scale=1, width=0.005
    )

    def frame_fn(sim: BoundarySimulation) -> None:
        """Called by sim.run at the end of each dt_frame; mutates circles+quiv."""
        # update circle centers
        for c, pos in zip(circles, sim.system.positions):
            c.center = pos
        # update quiver
        quiv.set_offsets(sim.system.positions)
        quiv.set_UVC(sim.system.velocities[:,0],
                     sim.system.velocities[:,1])

    def _update(frame_idx: int):
        # advance exactly one frame's worth of time, then redraw
        sim.run(n_steps=1, fn=frame_fn)
        return circles + [quiv]

    anim = animation.FuncAnimation(
        fig, _update,
        frames=range(n_frames),
        interval=interval,
        blit=True
    )
    return anim

# 1) build your system + sim
system = HardDiskSystem(
    box_size=10.0,
    particle_radius=0.3,
    n_particles=64,
    periodic_boundary=False
)
sim = BoundarySimulation(system, debug=False)
# choose a frame‐time
dt_frame = 0.01

# 2) kick off the animation
ani = animate_simulation(
    sim        = sim,
    n_frames   = 500,
    dt_frame   = dt_frame,
    interval   = 20,     # ms between GUI frames
    figsize    = (6,6)
)

# 3) Save to GIF using the Pillow writer
ani.save(
    'hard_disk_simulation.gif',
    writer='pillow',
    fps=int(1000/20),   # match your interval → here 50 fps
    dpi=80
)

plt.close()