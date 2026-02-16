import roboticstoolbox as rtb
from spatialmath import SE3, Twist3
from math import pi
import numpy as np

# Panda parameters from https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
def forwardKinematicsPoESpace(q):
    # Define the space twists (given in space frame) joints 1-7
    S = [
        Twist3.UnitRevolute([0, 0, 1], [0, 0, 0]),
        Twist3.UnitRevolute([0, 1, 0], [0, 0, 0.333]),
        Twist3.UnitRevolute([0, 0, 1], [0, 0, 0.649]),
        Twist3.UnitRevolute([0, -1, 0], [0.0825, 0, 0.649]),
        Twist3.UnitRevolute([0, 0, 1], [0, 0, 1.033]),
        Twist3.UnitRevolute([0, -1, 0], [0, 0, 1.033]),
        Twist3.UnitRevolute([0, 0, -1], [0.088, 0, 1.033])
    ]
    
    # Initial transformation matrix for Panda at home configuration
    M = SE3(0.088, 0, 0.926) * SE3.Rx(pi)
    
    # Compute forward kinematics using PoE in space form
    T = SE3()
    for i in range(7):
        T = T @ S[i].exp(q[i])
    T = T @ M
    
    return T

def forwardKinematicsPoEBody(q):
    # Define the body twists (given in body frame) joints 1-7
    B = [
        Twist3.UnitRevolute([0, 0, -1], [-0.088, 0, 0.593]),
        Twist3.UnitRevolute([0, -1, 0], [-0.088, 0, 0.593]),
        Twist3.UnitRevolute([0, 0, -1], [-0.088, 0, 0.277]),
        Twist3.UnitRevolute([0, 1, 0], [-0.0055, 0, 0.277]),
        Twist3.UnitRevolute([0, 0, -1], [-0.088, 0, -0.107]),
        Twist3.UnitRevolute([0, 1, 0], [-0.088, 0, -0.107]),
        Twist3.UnitRevolute([0, 0, 1], [0, 0, 0]),
    ]
    
    # Initial transformation matrix for Panda at home configuration
    M = SE3(0.088, 0, 0.926) * SE3.Rx(pi)
    
    # Compute forward kinematics using PoE in body form
    T = M
    for i in range(7):
        T = T @ B[i].exp(q[i])
    
    return T

def evalManufacturingTolerances():
    # Define the original robot and the modified one
    original_robot = rtb.models.DH.Panda()
    modified_robot = rtb.models.DH.Panda()
    
    # Modify the DH parameters
    modified_robot.links[0].d = 0.383  # Modify d1
    modified_robot.links[2].alpha = 1.5807  # Modify alpha3
    
    # joint limits
    num_samples = 1000
    lower_limits = original_robot.qlim[0, :]  # lower limits for each joint
    upper_limits = original_robot.qlim[1, :]  # upper limits for each joint
    
    # Sample random joint configurations within limits
    joint_configs = np.random.uniform(low=lower_limits, high=upper_limits, size=(num_samples, 7))

    # Lists to store individual position errors in x, y, z
    x_errors = []
    y_errors = []
    z_errors = []
    
    # Lists to store orientation errors in roll, pitch, yaw
    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    
    for q in joint_configs:
        original_fk = original_robot.fkine(q)
        modified_fk = modified_robot.fkine(q)
        
        # Compute the error in position (Euclidean distance)
        position_error_vector = original_fk.t - modified_fk.t

        # Split into x, y, z components
        x_error = position_error_vector[0]
        y_error = position_error_vector[1]
        z_error = position_error_vector[2]
        
        # Convert rotation matrices to Euler angles (roll, pitch, yaw)
        # Note: The order of rotation can vary. Here, we use the ZYX convention
        y_original, p_original, r_original = original_fk.eul()
        y_modified, p_modified, r_modified = modified_fk.eul()
        
        # Compute the individual roll, pitch, yaw errors
        roll_error = r_original - r_modified
        pitch_error = p_original - p_modified
        yaw_error = y_original - y_modified
        
        # Append component errors to lists
        x_errors.append(np.abs(x_error))
        y_errors.append(np.abs(y_error))
        z_errors.append(np.abs(z_error))

        # Append orientation component errors to lists
        roll_errors.append(np.abs(roll_error))
        pitch_errors.append(np.abs(pitch_error))
        yaw_errors.append(np.abs(yaw_error))

    # Error statistics for individual x, y, z errors
    x_mean = np.mean(x_errors)
    y_mean = np.mean(y_errors)
    z_mean = np.mean(z_errors)

    x_std = np.std(x_errors)
    y_std = np.std(y_errors)
    z_std = np.std(z_errors)

    x_max = np.max(x_errors)
    y_max = np.max(y_errors)
    z_max = np.max(z_errors)

    # Error statistics for roll, pitch, yaw errors
    roll_mean = np.mean(roll_errors)
    pitch_mean = np.mean(pitch_errors)
    yaw_mean = np.mean(yaw_errors)

    roll_std = np.std(roll_errors)
    pitch_std = np.std(pitch_errors)
    yaw_std = np.std(yaw_errors)

    roll_max = np.max(roll_errors)
    pitch_max = np.max(pitch_errors)
    yaw_max = np.max(yaw_errors)
    
    # Print individual component errors
    print(f"X error - Mean: {x_mean:.6f} m, Std: {x_std:.6f} m, Max: {x_max:.6f} m")
    print(f"Y error - Mean: {y_mean:.6f} m, Std: {y_std:.6f} m, Max: {y_max:.6f} m")
    print(f"Z error - Mean: {z_mean:.6f} m, Std: {z_std:.6f} m, Max: {z_max:.6f} m")

    # Print orientation component errors
    print(f"Roll error - Mean: {roll_mean:.6f} rad, Std: {roll_std:.6f} rad, Max: {roll_max:.6f} rad")
    print(f"Pitch error - Mean: {pitch_mean:.6f} rad, Std: {pitch_std:.6f} rad, Max: {pitch_max:.6f} rad")
    print(f"Yaw error - Mean: {yaw_mean:.6f} rad, Std: {yaw_std:.6f} rad, Max: {yaw_max:.6f} rad")

    return (x_mean, x_std, x_max, y_mean, y_std, y_max, z_mean, z_std, z_max,
            roll_mean, roll_std, roll_max, pitch_mean, pitch_std, pitch_max, yaw_mean, yaw_std, yaw_max)


# Load the robot model
robot = rtb.models.DH.Panda()
robot.tool = SE3() # we remove the tool offset as we are only interesed in the FK to the end-effector
robot.qz = [1, 1, -1, 1, -1, 1, -1]

# Q1.1
poe = forwardKinematicsPoESpace(robot.qz)
print("\n FK POE space", poe)

# Q1.2
poe_b = forwardKinematicsPoEBody(robot.qz)
print("\n FK POE body", poe_b)

# You can compare your implementation of the forward kinematics with the one from prtb
# test with multiple valid robot configurations
dhres = robot.fkine(robot.qz)
print("\n FK DH",dhres)

# Q1.3
evalManufacturingTolerances()