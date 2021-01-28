import numpy as np
import scipy
import matplotlib.pyplot as plt
import sympy

# -*- coding: utf-8 -*-

# Definition of microphone co-ordinates in 3D
h1 = [0, 0, 0]
h2 = [0, 10, 0]
h3 = [10, 10, 0]
h4 = [10, 0, 0]

# To plot the rectangle in 2D space
h1d = [h1[0], h1[1]]
h2d = [h2[0], h2[1]]
h3d = [h3[0], h3[1]]
h4d = [h4[0], h4[1]]

H2D = np.array([h1d, h2d, h3d, h4d])

# Define variables delta t, the error in time of arrival
Dt1 = [0.00]
Dt2 = [0.000]
Dt3 = [0.0]
Dt4 = [0.000]
DT = np.array([Dt1, Dt2, Dt3, Dt4])

# Put microphone co-ordinates in a matrix, H
H = np.array([h1, h2, h3, h4])

# Set variables
temp_assume = 20
rel_humid = 85  # As a percentage
v_sound = np.sqrt(1.402 * 8310 * temp_assume / 28.966)  # Speed of sound
coeff = 0.02  # Attenuation coefficient. Typical value at 20 deg and 10kHz and 20% RH is 0.065 from Bohn paper. Or it is 0.024 at 60% RH. 80% is 0.02
db_of_sound_source = 85
v_wind = 8  # Wind speed
temp_deg = 30
degrees = 45  # Set degrees of wind direction
dir_wind_rad = degrees * np.pi / 180  # Direction of wind in radians
dir_wind = np.array([1, 1, 0])
dir_wind = dir_wind / np.linalg.norm(dir_wind)
a_xrange = 21  # Maximum range in x for position of sound source, a
a_xmin = 5 - a_xrange / 2
a_yrange = 21  # Maximum range in y for position of sound source, a
a_ymin = 5 - a_yrange / 2
a_zrange = 0
a_zmin = 0
dc = 10  # Speed of sound error

increment = 1  # MUST TURN OUT TO BE RATIONAL
i_loop = int(a_xrange / increment)
j_loop = int(a_yrange / increment)


# no_steps_x = int(max_a_x*2/increment + 1)
# no_steps_y = int(max_a_y*2/increment + 1)
#
# x_range = np.arange((5 - max_a_x), (5+max_a_x) , increment)
# y_range = np.arange((math.sqrt(75)/2 - max_a_y), (math.sqrt(75)/2 + max_a_y), increment)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'





# I think this may be wrong but it looks good


# Trying to do wind direction
# We want to change the speed of sound, depending on which sensor the sound is travelling to AS WELL AS the position of the sound source.
# So would want a loop, for each position a then you get a new error in speed of sound, for each sensor you get an error. Nice




# Now we want to calculate some sort of error integral, to add up all the error in a box and then we can get some ranking of how important the parameter is.















# plot_sos_time_total(H,DT,dc,v_sound,a_xrange,a_yrange)

# plot_wind_mic2_4(H, dir_wind, v_wind, a_xrange, a_yrange)

# plot_temp(H,temp_deg,a_xrange,a_yrange)
#
# plot_humid(H,temp_deg,rel_humid,a_xrange,a_yrange)
#
# plot_2D_nominal(H, v_sound, a_xrange, a_yrange, a_z=2)
#
# plot_pair_sourceHeight(h2d,h4d,10,1)

# plot_SNR(coeff,db_of_sound_source)
