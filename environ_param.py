from error_model import R
from enviro_calc import humid, sos
import matplotlib.pyplot as plt
import numpy as np

H = np.array([0,0,0],dtype=float)
number_points = 200
range_points = 60
z = 5

# Set up the microphone position as static
H = np.array([0, 0, 0])

# Set up x, y and z for positions of a (sound source)
x = np.linspace(0, range_points, number_points).reshape(-1, 1)
y = np.zeros_like(x)
z = np.zeros_like(x)

# Create an (number_points x 3) array of the 'a' positions
a = np.concatenate((x, y, z), axis=1)


def distance(a, H):
    """

    :param a: 1 x 3 array
    :param H: i x 3 array for the 1D case
    :return: 1 x i array of distance values
    """
    z = H - a
    return np.sqrt(np.sum(np.square(z), axis=1))#.reshape(-1, 1)


def error_2D_in_1D(v_sound, height=0):
    """
    Calculates and plots the error in ToA to a microphone as a function of vertical height error, z
    :param v_sound:
    :param z:
    :return:
    """
    # Set new z value for the soudn source location
    height = height * np.ones_like(x)
    b = np.concatenate((x, y, height), axis=1)

    # Calculate ToA for a 3D model, a (number_points x 1) array
    ToF_3D = distance(a, H) / v_sound

    # ToA for 1D case is just the x distance
    ToF_2D = a[:, 0] / v_sound

    # Subtract two to obtain error
    ToF_error = ToF_3D - ToF_2D

    Distance_error = ToF_error * v_sound

    return ToF_error

# Next error in temperature differences. Easy way to do this is I think is to have a standard method for calculating change in speed of sound
# Can I do this with only 1 microphone --> don't think so, need two

def error_SoS_in_1D(dc, v_sound):

    ToF_predicted = distance(a, H) / v_sound

    ToF_actual = distance(a, H) / (v_sound + dc)

    ToF_error = ToF_actual - ToF_predicted

    Distance_error = ToF_error * (v_sound + dc)     # Should this be + dc or not??

    return ToF_error, Distance_error


def error_humidty_in_1D(rel_humidity, temp_deg_assumed=20, rel_humidity_assumed=85):

    v_sound_actual = sos(temp_deg_assumed, rel_humidity)

    v_sound_predicted = sos(temp_deg_assumed, rel_humidity_assumed)

    dc = v_sound_actual - v_sound_predicted

    return error_SoS_in_1D(dc, v_sound_predicted)


def error_temp_in_1D(temp_deg, temp_deg_assumed=20, rel_humidity=85):
    v_sound_actual = sos(temp_deg, rel_humidity)

    v_sound_predicted = sos(temp_deg_assumed, rel_humidity)

    dc = v_sound_actual - v_sound_predicted

    return error_SoS_in_1D(dc, v_sound_predicted)


def error_wind_in_1D(wind, temp_deg_assume=20, rel_humidity=85):

    v_sound_predicted = sos(temp_deg_assume, rel_humidity)

    return error_SoS_in_1D(wind, v_sound_predicted)


def plot_variable_1D(variable_list, variable='temperature'):
    variable_dict = {'temperature': error_temp_in_1D, 'wind': error_wind_in_1D, 'humidity': error_humidty_in_1D}
    label_dict = {'temperature': "\u00b0C", 'wind': "m/s", 'humidity': "% RH"}

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for item in variable_list:
        ToF_error, Distance_error = variable_dict[variable](item)
        ax1.plot(x, ToF_error, label="%s %s" % (item, label_dict[variable]))
        ax2.plot(x, Distance_error, label="%s %s" % (item, label_dict[variable]))


    ax1.set_xlabel("x (m)", fontsize=16)
    ax1.set_ylabel("error (s)", fontsize=16)
    ax2.set_ylabel("error (m)", fontsize=16)
    fig.suptitle("%s" % variable)

    plt.legend()

    plt.show()

temp = [10, 15, 20, 25, 30]
wind = [-5, -2 - 1, 1, 2, 5, 10]
humidity = [0, 20, 50, 85]
z = [0, 1, 2, 5]

# plot_variable_1D(wind, variable='wind')
# plot_variable_1D(temp, variable='temperature')
# plot_variable_1D(humidity, variable='humidity')

# Things affecting SNR
"""
    - temperature
    - humidity
    - environment (leaf etc)
    - strength of signal
    - directionality of signal
"""

