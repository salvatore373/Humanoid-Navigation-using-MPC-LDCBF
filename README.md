
# Humanoid Navigation using Control Barrier Functions
Authors: [Salvatore Michele Rago](https://github.com/salvatore373), [Damiano Imola](https://github.com/damianoimola), [Eugenio Bugli](https://github.com/EugenioBugli).

Humanoids are robots designed to navigate in environments structured for humans. Their dynamics is very complex and has to be taken into account for path planning and gait control, which in turn must be carried out in real time. Hence, they are usually decoupled to reduce the computational load. The novelty of this project consists in describing the problem in an efficient way that allows to solve simultaneously path and gait planning.
This project is based on "[Real-Time Safe Bipedal Robot Navigation using Linear Discrete Control Barrier Functions](https://arxiv.org/abs/2411.03619)" by Peng et al. However, the proposed solution is unfeasible with the specified constraints, it does not take into account for stability and equilibrium constraints, and it does not provide infomration on how the orientation and turning rate are precomputed. In our implementation, we provide our solution to those problems.

A complete description of the framework can be found in the [Report](https://github.com/salvatore373/Humanoid-Navigation-using-MPC-LDCBF/blob/main/Report/main.pdf).

## 3D-LIP model with Heading Angles
If the full dynamic model of the humanoid is used to simulate its motion, it becomes computationally impossible to perform joint path and gait planning, due to its high dimensionality and non-linearity. Therefore, a simplifying model must be used. For this scope the "3D-LIP Model with Heading Angle", which describes the discrete dynamics of the Center of Mass (CoM) similarly to the one of an inverted pendulum in three dimensions.

## LIP-MPC: Gait planning with Model Predictive Control
The LIP dynamics is used as a model of the process inside a Model Predictive Control (MPC) scheme. Since the imposed constraints are non-linear, a non-linear MPC scheme would be required to solve the optimization problem. It is very expensive from a computational point of view and would compromise real-time performance. To overcome this non-linear representation, the values of $\theta$ and $\omega$ are not determined by the MPC optimization, but they are precomputed. In this way, all the constraints imposed in the MPC become linear, and a linear MPC can be used.

## Simulations
The humanoid is represented as an isoscele triangle, whose barycenter is coincident with the robot's CoM, and their sagittal axis are aligned. The humanoid is always required to go from the input initial pose to the user-defined goal position while avoiding all the obstacles populating the environment.

### Basic simulation
In this scenario, the robot has to avoid collisions with the 3 quasi-circular obstacles in the environment.

![simulation with custom LDCBF with circular obstacles](https://github.com/salvatore373/Humanoid-Navigation-using-MPC-LDCBF/blob/main/Assets/ReportResults/Simulation1CirclesDelta/animation.gif)

### Unknown environment
In this simulation, the robot moves in the environment without having an a-priori knowledge of the workspace geometry: the obstacles are sensed in real-time using LiDAR scans.

![simulation in an unknown environment](https://github.com/salvatore373/Humanoid-Navigation-using-MPC-LDCBF/blob/main/Assets/ReportResults/Simulation4UnkEnv/animation.gif)

### RRT* variation
The standard framework would not be able to reach the goal in this scenario, as the MPC gets stuck in a local minimum. Our extension of the original framework achieves goal attainment by following a sequence of sub-goals defined by RRT*.

![simulation with RRT extended framework](https://github.com/salvatore373/Humanoid-Navigation-using-MPC-LDCBF/blob/main/Assets/ReportResults/SimulationRRT/animation.gif)
