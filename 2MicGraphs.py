from enviro_calc import humid, sos
import matplotlib.pyplot as plt
from environ_param import distance
import numpy as np

# Set up 2 microphone array as static
H = np.array([[20,0,0],[40, 0, 0]],dtype=float)
number_points = 200
range_points = 60

# Set up x, y and z for positions of a (sound source)
x = np.linspace(0, range_points, number_points).reshape(-1, 1)
y = np.zeros_like(x)
z = np.zeros_like(x)

# Create an (number_points x 3) array of the 'a' positions
a = np.concatenate((x, y, z), axis=1)

def distance_multimic(a, H):
    """

    :param a: 1 x 3 array
    :param H: i x 3 array
    :return: 1 x i array of distance values
    """
    z = np.zeros((a.shape[0], H.shape[0]))
    for i in range(H.shape[0]):
        z[:, i] = np.sqrt(np.sum(np.square(H[i, :] - a), axis=1))
    return z


def error_2D_in_2D(height, temp_deg_assumed=20, rel_humidity=85):
    """
    Calculates and plots the error in ToA to a microphone as a function of vertical height error, z
    :param v_sound:
    :param z:
    :return:
    """
    v_sound_assumed = sos(temp_deg_assumed, rel_humidity)

    # Set new z value for the sound source locatins, b
    height = height * np.ones_like(x)
    b = np.concatenate((x, y, height), axis=1)

    # Calculate TDOA, as this forms the basis of the error
    TDoA_3D = np.subtract(distance_multimic(b, H)[:, 0], distance_multimic(b, H)[:, 1]) / v_sound_assumed
    print(b)

    # Obtain the predicted differential distance
    diff_distance_predicted = TDoA_3D * v_sound_assumed

    # Obtain the actual differential distance (in 2D plane) by considering the difference in distance of the HORIZONTAL aspects of a rather than b
    diff_distance_actual = np.subtract(distance_multimic(a, H)[:, 0], distance_multimic(a, H)[:, 1])

    diff_distance_error = diff_distance_actual - diff_distance_predicted

    return TDoA_3D, diff_distance_error

def error_SoS_in_2D(dc, v_sound_assumed):
    """
    TDoA: 1 x d where d is the no_points in a
    :param dc:
    :param v_sound_assumed:
    :return:
    """
    # Time difference of arrival between the first and second microphone using the ACTUAL speed of sound
    TDoA = np.subtract(distance_multimic(a, H)[:, 0], distance_multimic(a, H)[:, 1]) / (v_sound_assumed + dc)

    diff_distance_actual = TDoA * (v_sound_assumed + dc)

    diff_distance_predicted = TDoA * v_sound_assumed

    diff_distance_error = diff_distance_actual - diff_distance_predicted

    return TDoA, diff_distance_error

def plot_variable_2D(variable_list, variable='temperature'):
    variable_dict = {'temperature': error_temp_in_2D, 'wind': error_wind_in_2D, 'humidity': error_humidity_in_2D, '2D error': error_2D_in_2D}
    label_dict = {'temperature': "\u00b0C", 'wind': "m/s", 'humidity': "% RH", '2D error': 'm'}

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for item in variable_list:
        TDoA, Distance_error = variable_dict[variable](item)
        ax1.plot(x, TDoA, label="%s %s" % (item, label_dict[variable]))
        ax2.plot(x, Distance_error, label="%s %s" % (item, label_dict[variable]))


    ax1.set_xlabel("x (m)", fontsize=16)
    ax1.set_ylabel("TDoA (s)", fontsize=16)
    ax2.set_ylabel("error (m)", fontsize=16)

    # Plot on microphones
    ax1.plot(H[0,0],H[0,1], 'ro')
    ax1.plot(H[1,0],H[1,1],'ro')
    ax2.plot(H[0, 0], H[0, 1], 'ro')
    ax2.plot(H[1, 0], H[1, 1], 'ro')

    fig.suptitle("%s" % variable)

    ax1.legend()

    plt.show()

def error_temp_in_2D(temp_deg, temp_deg_assumed=20, rel_humidity=85):
    v_sound_actual = sos(temp_deg, rel_humidity)

    v_sound_assumed = sos(temp_deg_assumed, rel_humidity)

    dc = v_sound_actual - v_sound_assumed

    return error_SoS_in_2D(dc, v_sound_assumed)

def error_humidity_in_2D(rel_humidity, temp_deg_assumed=20, rel_humidity_assumed=85):

    v_sound_actual = sos(temp_deg_assumed, rel_humidity)

    v_sound_assumed = sos(temp_deg_assumed, rel_humidity_assumed)

    dc = v_sound_actual - v_sound_assumed

    return error_SoS_in_2D(dc, v_sound_assumed)


def error_wind_in_2D(wind, temp_deg_assume=20, rel_humidity=85):

    v_sound_assumed = sos(temp_deg_assume, rel_humidity)

    return error_SoS_in_2D(wind, v_sound_assumed)

temp = [10, 15, 20, 25, 30]
wind = [-5, -2 - 1, 1, 2, 5, 10]
humidity = [0, 20, 50, 85, 98]
z = [0, 1, 2, 5]

plot_variable_2D(temp, variable='temperature')
plot_variable_2D(z, variable='2D error')