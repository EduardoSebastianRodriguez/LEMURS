import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from Functions import get_cmap

# Load data
na_train   = 4
na_tests   = 64
dis        = 5
step_size  = 0.04
time       = 15.0
trajectory = np.load('TVS_trajectory_learned'+str(na_train)+str(na_tests)+'_LEMURS.npy')

# Define the meta data for the movie
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Time-Varying Swapping', artist='Matplotlib', comment='time-varying swapping animation')
writer   = FFMpegWriter(fps=15*na_tests, metadata=metadata)

# Initialize the movie
fig = plt.figure()

# Plot setup
colors = get_cmap(na_tests)
plt.rcParams.update({'font.size': 20})
traces = ()
marks  = ()
for k in range(na_tests):
    trace, = plt.plot([], [], color=colors(k), linewidth=1)
    mark,  = plt.plot([], [], color=colors(k), marker='o', markersize=12)
    traces = traces + (trace,)
    marks  = marks + (mark,)
    plt.plot(trajectory[0, 4 * na_tests + 2 * k], trajectory[0, 4 * na_tests + 2 * k + 1], color=colors(k), marker='x', markersize=28)
plt.xlabel('x $[$m$]$')
plt.ylabel('y $[$m$]$')
plt.xlim([-4.5, 4.5])
plt.ylim([-3.0 * (int(na_tests/4)-1) - 1.5 - 1.5, 3.0 * (int(na_tests/4)-1) + 1.5 + 1.5])

# Update the frames for the movie
with writer.saving(fig, "TVSS_animation"+str(na_train)+str(na_tests)+".mp4", 300):
    for i in range(int(time/step_size)):
        for k in range(na_tests):
            marks[k].set_data(trajectory[i, 2*k], trajectory[i, 2*k+1])
            if i < dis:
                traces[k].set_data(trajectory[:i+1, 2*k], trajectory[:i+1, 2*k+1])
            else:
                traces[k].set_data(trajectory[i-dis:i+1, 2*k], trajectory[i-dis:i+1, 2*k+1])
            writer.grab_frame()