import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from Functions import get_cmap

# Load data
na_train   = 4
na_tests   = 64
dis        = 5
step_size  = 0.04
time       = 6.0
trajectory = np.load('F_trajectory_learned'+str(na_train)+str(na_tests)+'_LEMURS.npy')

# Define the meta data for the movie
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Flocking', artist='Matplotlib', comment='flocking animation')
writer   = FFMpegWriter(fps=15*na_tests, metadata=metadata)

# Initialize the movie
fig = plt.figure()

# Plot setup
plt.rcParams.update({'font.size': 20})
traces = ()
marks  = ()
for k in range(na_tests):
    trace, = plt.plot([], [], color='k', linewidth=1)
    mark,  = plt.plot([], [], color='b', marker='o', markersize=12)
    traces = traces + (trace,)
    marks  = marks + (mark,)
plt.plot(trajectory[0, 4 * na_tests], trajectory[0, 4 * na_tests + 1], color='r', marker='x', markersize=28)
plt.xlabel('x $[$m$]$')
plt.ylabel('y $[$m$]$')
plt.xlim([-2.0, 5.0])
plt.ylim([-2.0, 5.0])

# Update the frames for the movie
with writer.saving(fig, "F_animation"+str(na_train)+str(na_tests)+".mp4", 300):
    for i in range(int(time/step_size)):
        for k in range(na_tests):
            if i == int(time/step_size/4*2):
                plt.xlim([-1.0, 1.0])
                plt.ylim([-1.0, 1.0])
            if i == int(time/step_size/8*6):
                plt.xlim([-0.5, 0.5])
                plt.ylim([-0.5, 0.5])
            marks[k].set_data(trajectory[i, 2*k], trajectory[i, 2*k+1])
            if i < dis:
                traces[k].set_data(trajectory[:i+1, 2*k], trajectory[:i+1, 2*k+1])
            else:
                traces[k].set_data(trajectory[i-dis:i+1, 2*k], trajectory[i-dis:i+1, 2*k+1])
            writer.grab_frame()