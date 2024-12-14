import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation


# ===== CONSTANTS =====
STEP_SIZE = 6
number_of_shadows = 10
foot_distance = 3
number_of_footsteps = 20


# ===== UTILITY FUNCTIONS =====
def draw_robot(pose, alpha=1.0):
    robot = plt.Circle((pose[0], pose[1]), 2, color='tomato', fill=True, linewidth=2, alpha=alpha)
    ax.add_patch(robot)

def draw_tick(pose, alpha=1.0):
    tick_length = 1
    tick_x = [pose[0], pose[0] + tick_length * np.cos(pose[2])]
    tick_y = [pose[1], pose[1] + tick_length * np.sin(pose[2])]
    plt.plot(tick_x, tick_y, color='black', linewidth=2, alpha=alpha)


# ===== INITIAL CONFIGURATION =====
# random initial position
start = np.random.randint(-10, 10, (3, 1))
start = np.array([start[0], start[1], np.deg2rad(np.rad2deg(start[2]) % 360)])

# random goal position
goal = np.random.randint(-10, 10, (3, 1))
goal = np.array([goal[0], goal[1], np.deg2rad(np.rad2deg(goal[2]) % 360)])

# initial left foot position
left_foot = np.array([
    start[0] - foot_distance * np.sin(start[2]),
    start[1] + foot_distance * np.cos(start[2])
])

# initial right foot position
right_foot = np.array([
    start[0] + foot_distance * np.sin(start[2]),
    start[1] - foot_distance * np.cos(start[2])
])


# container of previous values
com_pose_history = [start]
left_foot_history = [left_foot]
right_foot_history = [right_foot]


last_seen_frame = -1
def update(frame):
    global start, left_foot, right_foot, last_seen_frame
    print("### FRAME", frame)

    # explanation of such control:
    # https://stackoverflow.com/questions/74252467/why-when-doing-animations-with-matplotlib-frame-0-appears-several-times
    if last_seen_frame != frame:
        last_seen_frame = frame
    else: return

    # clear current axis
    plt.cla()

    step_size = STEP_SIZE/2 if frame==0 else STEP_SIZE

    # goal marker
    plt.scatter(goal[0], goal[1], marker="o", color="royalblue", label='goal', s=300)

    # draw last 10 CoMs
    for idx, com in enumerate(list(reversed(com_pose_history))[:min(len(list(com_pose_history)), number_of_shadows)]):
        # start marker
        draw_robot(com, alpha=1.0-idx*1/number_of_shadows)
        draw_tick(com, alpha=1.0-idx*1/number_of_shadows)

    # draw last 10 left footsteps
    for idx, left in enumerate(list(reversed(left_foot_history))[:min(len(list(left_foot_history)), number_of_shadows//2)]):
        # condition to plot in legend only one foot
        if idx == 0: plt.scatter(left[0], left[1], marker="o", color="green", label='left foot', s=20, alpha=1.0-idx*1/number_of_shadows/2)
        else: plt.scatter(left[0], left[1], marker="o", color="green", s=20, alpha=1.0-idx*1/number_of_shadows/2)

    # draw last 10 right footsteps
    for idx, right in enumerate(list(reversed(right_foot_history))[:min(len(list(right_foot_history)), number_of_shadows//2)]):
        # condition to plot in legend only one foot
        if idx == 0: plt.scatter(right[0], right[1], marker="o", color="lightgreen", label='right foot', s=20, alpha=1.0-idx*1/number_of_shadows/2)
        else: plt.scatter(right[0], right[1], marker="o", color="lightgreen", s=20, alpha=1.0-idx*1/number_of_shadows/2)


    # ===== COMPUTING NEXT VALUES =====
    start = np.array([
        start[0] + step_size / 2 * np.cos(start[2]),
        start[1] + step_size / 2 * np.sin(start[2]),
        start[2]
    ])
    com_pose_history.append(start)

    foot = 0 if frame % 2 == 0 else 1

    if foot == 0:
        print("LEFT")
        left_foot = np.array([
            left_foot[0] + step_size * np.cos(start[2]),
            left_foot[1] + step_size * np.sin(start[2])
        ])
        left_foot_history.append(left_foot)
    else:
        print("RIGHT")
        right_foot = np.array([
            right_foot[0] + step_size * np.cos(start[2]),
            right_foot[1] + step_size * np.sin(start[2])
        ])
        right_foot_history.append(right_foot)

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.subplots_adjust(right=0.75)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=1, fancybox=True, shadow=False, fontsize="13")


fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=20, interval=200)
ani.save('../Assets/Animations/humanoid_2d_animation.gif', writer='ffmpeg')
plt.show()


