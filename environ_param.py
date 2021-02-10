from error_model import R
from enviro_calc import humid, sos
import matplotlib.pyplot as plt
import numpy as np


def error_2D_in_1D(v_sound, z=0):
    """
    Calculates and plots the error in ToA to a microphone as a function of vertical height error, z
    :param v_sound:
    :param z:
    :return:
    """

    H = np.array([0, 0, 0])

    x = np.linspace(0, 60, 200)

    a = np.array([x, 0, z])

    ToA_error = (R(a, H) - R(a, H, model="2D")) / v_sound  # This should be a 1x1 array

    return x, ToA_error


# Next error in temperature differences. Easy way to do this is I think is to have a standard method for calculating change in speed of sound
# Can I do this with only 1 microphone --> don't think so, need two

def error_SoS_in_1D(dc, v_sound):
    H = np.array([0, 0, 0])

    x = np.linspace(0, 60, 200)

    a = np.array([x, 0, 0])

    ToF_predicted = R(a, H) / v_sound

    ToF_actual = R(a, H) / (v_sound + dc)

    ToF_error = ToF_actual - ToF_predicted

    distance_error = ToF_error * (v_sound + dc)

    return x, distance_error


def error_humidty_in_1D(rel_humidity, temp_deg_assumed=20, rel_humidity_assumed=85):

    v_sound_actual = sos(temp_deg_assumed, rel_humidity)

    v_sound_predicted = sos(temp_deg_assumed, rel_humidity_assumed)

    dc = v_sound_actual - v_sound_predicted

    x, distance_error = error_SoS_in_1D(dc, v_sound_predicted)

    return x, distance_error


def error_temp_in_1D(temp_deg, temp_deg_assumed=20, rel_humidity=85):
    v_sound_actual = sos(temp_deg, rel_humidity)

    v_sound_predicted = sos(temp_deg_assumed, rel_humidity)

    dc = v_sound_actual - v_sound_predicted

    x, distance_error = error_SoS_in_1D(dc, v_sound_predicted)

    return x, distance_error


def error_wind_in_1D(wind, temp_deg_assume=20, rel_humidity=85):
    #
    # direction = (a - H)/np.linalg.norm(a - H)
    #
    # dc = np.dot(wind, direction)

    v_sound = sos(temp_deg_assume, rel_humidity)

    x, distance_error = error_SoS_in_1D(wind, v_sound)

    return x, distance_error


def plot_variable_1D(variable_list, variable='temperature'):
    variable_dict = {'temperature': error_temp_in_1D, 'wind': error_wind_in_1D, 'humidity': error_humidty_in_1D}

    plt.figure()

    for item in variable_list:
        x, distance_error = variable_dict[variable](item)
        plt.plot(x, distance_error, label="%s" % item)

    plt.xlabel("x (m)", fontsize=16)
    plt.ylabel("error (m)", fontsize=16)

    plt.legend()

    plt.show()

temp = [10, 15, 20, 25, 30]
wind = [-5, -2 - 1, 1, 2, 5, 10]
humidity = [0, 20, 50, 85]
z = [0, 1, 2, 5]

plot_variable(wind, variable='wind')
plot_variable(temp)
plot_variable(humidity, variable='humidity')

# Things affecting SNR
"""
    - temperature
    - humidity
    - environment (leaf etc)
    - strength of signal
    - directionality of signal
"""