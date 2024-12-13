import numpy as np
import matplotlib.pyplot as plt
import math


start = np.random.randint(-10, 10, (3, 1))
start = np.array([start[0], start[1], np.rad2deg(start[2])%360])

goal = np.random.randint(-10, 10, (3, 1))
goal = np.array([goal[0], goal[1], np.rad2deg(goal[2])%360])

print(start, np.deg2rad(start[2]))


foot_distance = 1
left_foot = np.array([
    start[0] + foot_distance*np.cos(np.deg2rad(start[2])),
    start[1] + foot_distance*np.sin(np.deg2rad(start[2]))
])
right_foot = np.array([
    start[0] + -foot_distance*np.cos(np.deg2rad(start[2])),
    start[1] + -foot_distance*np.sin(np.deg2rad(start[2]))
])


# start marker
plt.scatter(start[0], start[1], marker="o", color="tomato", label='start', s=300)
# start marker orientation left foot
plt.scatter(left_foot[0], left_foot[1], marker="o", color="green", label='left foot', s=100, alpha=0.7)
# start marker right foot
plt.scatter(right_foot[0], right_foot[1], marker="o", color="lightgreen", label='left foot', s=100, alpha=0.7)
# start marker orientation
plt.scatter(start[0], start[1], marker=(2, 0, start[2]), color="black", s=300)
# goal marker
plt.scatter(goal[0], goal[1], marker="o", color="royalblue", label='goal', s=300)



plt.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=1, fancybox=True, shadow=False, fontsize="13")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.subplots_adjust(right=0.75)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()