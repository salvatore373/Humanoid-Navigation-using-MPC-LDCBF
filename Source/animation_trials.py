import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# starting from the kinematic model you animate the triangle by updating the position of each vertices 
fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)

v = 0.1
s = [0.4 + v*np.cos(2*np.pi/t), 0.6 +v*np.sin(2*np.pi/t)]
s1 = [1,1] + s
s2 = [2,2] + s
s3 = [3,3] + s
print(s1)
scat1 = ax.scatter(s1[2][0], s1[3][0], c="b", s=1)
scat2 = ax.scatter(s2[2][0], s2[3][0], c="r", s=1)
scat3 = ax.scatter(s3[2][0], s3[3][0], c="g", s=1)
ax.set(xlim=[0, 1], ylim=[-4, 1], xlabel='Time [s]', ylabel='Z [m]')

def update(frame):
    # for each frame, update the data stored on each artist.
    scat1.set_offsets(np.stack([s1[2][:frame], s1[3][:frame],]).T)
    scat2.set_offsets(np.stack([s2[2][:frame], s2[3][:frame],]).T)
    scat3.set_offsets(np.stack([s3[2][:frame], s3[3][:frame],]).T)
    return [scat1, scat2, scat3]


ani = animation.FuncAnimation(fig=fig, func=update, frames=60, interval=30)
plt.show()