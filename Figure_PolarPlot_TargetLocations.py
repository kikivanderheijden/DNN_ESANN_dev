
# script to create a polar plot of locations tested in study

# import libraries and packages
dirfiles = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN'

import numpy as np
import matplotlib.pyplot as plt

azimuthrange = np.arange(0,360,10)


fig = plt.figure()
ax = plt.axes(polar=True)
r = np.ones(len(azimuthrange))
theta = azimuthrange
sizemark = 50
colors = theta
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.scatter(np.radians(theta), r, s = sizemark, c = colors, cmap = 'hsv')
ax.grid(False)
ax.set_yticks([])
plt.savefig(dirfiles+'/plot_polar_alltargetlocs.png')

