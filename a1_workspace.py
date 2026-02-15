# %%
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import spatialgeometry as sg
import matplotlib.pyplot as plt
from math import pi

# %%
robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(a=1, qlim=[0, pi/2]),
        rtb.PrismaticDH(theta=pi/2, qlim=[0.5, 1.5]),
        rtb.RevoluteDH(a=1, qlim=[-pi/2, pi/2]),
    ], name="myRobot")

# %%
def sample_workspace(robot, num_samples) -> np.ndarray:
    # Create a grid of joint values
    q1_vals = np.linspace(robot.links[0].qlim[0], robot.links[0].qlim[1], num_samples)
    q2_vals = np.linspace(robot.links[1].qlim[0], robot.links[1].qlim[1], num_samples)
    q3_vals = np.linspace(robot.links[2].qlim[0], robot.links[2].qlim[1], num_samples)

    # Store the end-effector positions
    workspace = []

    # Loop through each combination of joint values
    for q1 in q1_vals:
        for q2 in q2_vals:
            for q3 in q3_vals:
                q = [q1, q2, q3]  # Joint configuration
                # Forward kinematics to compute end-effector pose
                T = robot.fkine(q)
                # Get the position (x, y, z) from the transformation matrix
                pos = T.t
                workspace.append(pos)

    workspace = np.array(workspace)
    
    return workspace

def plot_workspace(workspace, elev=30, azim=30, ax=None, zoom_factor=1.5):
    fig = plt.figure()
    if not ax:
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(workspace[:,0], workspace[:,1], workspace[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.view_init(elev=elev, azim=azim)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    ax.set_xlim([xlim[0] * zoom_factor - 1.5, xlim[1] * zoom_factor - 1.5])
    ax.set_ylim([ylim[0] * zoom_factor, ylim[1] * zoom_factor])
    ax.set_zlim([zlim[0] * zoom_factor, zlim[1] * zoom_factor])

def plot_bounding_box(workspace, ax=None):
    fig = plt.figure()
    if not ax:
        ax = fig.add_subplot(111, projection='3d')
    
    # Calculate the min and max values for the bounding box
    x_min, x_max = np.min(workspace[:, 0]), np.max(workspace[:, 0])
    y_min, y_max = np.min(workspace[:, 1]), np.max(workspace[:, 1])
    z_min, z_max = np.min(workspace[:, 2]), np.max(workspace[:, 2])

    # Define the vertices of the bounding box
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],  # Bottom face
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],  # Top face
    ])

    # List of edges to connect the corners
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    for edge in edges:
        ax.plot([corners[edge[0], 0], corners[edge[1], 0]],
                [corners[edge[0], 1], corners[edge[1], 1]],
                [corners[edge[0], 2], corners[edge[1], 2]], color='b')

# %%
q = [0, 0.5, 0]
ax = robot.plot(q)

workspace = sample_workspace(robot, num_samples=10)

plot_workspace(workspace, ax=ax.ax, elev=45, azim=320, zoom_factor=3)
plot_bounding_box(workspace, ax=ax.ax)
ax.fig.savefig('workspace.jpg', dpi=300)
ax.fig


