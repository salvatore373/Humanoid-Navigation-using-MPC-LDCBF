import numpy as np
import matplotlib.pyplot as plt
import math

# needed for later
fig = plt.gcf()
ax = fig.gca()

# ===== CONSTANTS =====
step_size = 3
foot_distance = 1

# random initial position
start = np.random.randint(-10, 10, (3, 1))
start = np.array([start[0], start[1], np.deg2rad(np.rad2deg(start[2])%360)])

# random goal position
goal = np.random.randint(-10, 10, (3, 1))
goal = np.array([goal[0], goal[1], np.deg2rad(np.rad2deg(goal[2])%360)])

# initial left foot position
left_foot = np.array([
    start[0] - foot_distance*np.sin(start[2]),
    start[1] + foot_distance*np.cos(start[2])
])

# initial right foot position
right_foot = np.array([
    start[0] + foot_distance*np.sin(start[2]),
    start[1] - foot_distance*np.cos(start[2])
])

def draw_robot(pose, alpha=1.0):
    robot = plt.Circle((pose[0], pose[1]), 1, color='tomato', fill=True, linewidth=2, alpha=alpha)
    ax.add_patch(robot)

def draw_tick(pose, alpha=1.0):
    tick_length = 1
    tick_x = [pose[0], pose[0] + tick_length * np.cos(pose[2])]
    tick_y = [pose[1], pose[1] + tick_length * np.sin(pose[2])]
    ax.plot(tick_x, tick_y, color='black', linewidth=2, alpha=alpha)

# start marker
draw_robot(start)
draw_tick(start)

# start marker orientation left foot
plt.scatter(left_foot[0], left_foot[1], marker="o", color="green", label='left foot', s=100, alpha=0.7)

# start marker right foot
plt.scatter(right_foot[0], right_foot[1], marker="o", color="lightgreen", label='right foot', s=100, alpha=0.7)

# goal marker
plt.scatter(goal[0], goal[1], marker="o", color="royalblue", label='goal', s=300)

for i in range(10):
    foot = 0 if i%2==0 else 1

    # FIXME: make it a one-line expression
    if i == 0:
        step_size = step_size/2
    else:
        step_size = 3

    start = np.array([
        start[0] + step_size / 2 * np.cos(start[2]),
        start[1] + step_size / 2 * np.sin(start[2]),
        start[2]
    ])

    draw_robot(start, alpha=1.0-i*0.1)
    draw_tick(start, alpha=1.0-i*0.1)

    if foot == 0:
        left_foot = np.array([
            left_foot[0] + step_size * np.cos(start[2]),
            left_foot[1] + step_size * np.sin(start[2])
        ])
        plt.scatter(left_foot[0], left_foot[1], marker="o", color="green", s=100, alpha=0.7)
    else:
        right_foot = np.array([
            right_foot[0] + step_size * np.cos(start[2]),
            right_foot[1] + step_size * np.sin(start[2])
        ])
        plt.scatter(right_foot[0], right_foot[1], marker="o", color="lightgreen", s=100, alpha=0.7)



plt.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=1, fancybox=True, shadow=False, fontsize="13")
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.subplots_adjust(right=0.75)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()


