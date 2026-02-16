import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift
import numpy as np
import time

# Load a robot robot
robot1 = rtb.models.Panda()
robot2 = rtb.models.Panda()

# Init robot1 joint to the 'ready' joint angles
robot1.q = robot1.qr
# Init the base of robot1
robot1.base = sm.SE3(0.6, 0, 0) * sm.SE3.Rz(np.pi)
robot1.grippers[0].q = [0.01, 0.01]

# Load rod
rod = sg.Cylinder(0.01, 0.3, pose=robot1.fkine(robot1.q) * sm.SE3.Ry(np.pi/2) * sm.SE3(0, 0, 0.1))

# Init robot2 joint to the 'ready' joint angles
robot2.base = sm.SE3(-0.5, -0.1, 0)
robot2.grippers[0].q = [0.01, 0.01]
robot2.q = robot2.ikine_LM(robot2.base.inv() * robot1.fkine(robot1.q) * sm.SE3(0.2, 0, 0), q0=robot2.q).q

# Launch the simulator Swift
env = swift.Swift()
env.launch()
# add the robot to the environment
env.add(robot1)
env.add(robot2)
env.add(rod)

# This is our callback funciton from the sliders in Swift which set
# the joint angles of our robot to the value of the sliders
def set_joint(j, value):
    # Save the old joint angle
    old_q = robot1.q[j]
    robot1.q[j] = np.deg2rad(float(value))

    # Calculate the IK solution for robot2
    ik_solution = robot2.ikine_LM(robot2.base.inv() * robot1.fkine(robot1.q) * sm.SE3(0.2, 0, 0), q0=robot2.q)

    # If the IK solution is successful, update the robot pose
    if ik_solution.success:
        rod.T = robot1.fkine(robot1.q) * sm.SE3.Ry(np.pi/2) * sm.SE3(0, 0, 0.1)
        robot2.q = ik_solution.q
    else:
        robot1.q[j] = old_q

# Loop through each link in the robot and if it is a variable joint,
# add a slider to Swift to control it
j = 0
for link in robot1.links:
    if link.isjoint:

        # We use a lambda as the callback function from Swift
        # j=j is used to set the value of j rather than the variable j
        # We use the HTML unicode format for the degree sign in the unit arg
        env.add(
            swift.Slider(
                lambda x, j=j: set_joint(j, x,),
                min=np.round(np.rad2deg(link.qlim[0]), 2),
                max=np.round(np.rad2deg(link.qlim[1]), 2),
                step=1,
                value=np.round(np.rad2deg(robot1.q[j]), 2),
                desc="robot1 Joint " + str(j),
                unit="&#176;",
            )
        )

        j += 1


while True:
    # Update the environment with the new robot pose
    env.step(0)
    time.sleep(0.01)