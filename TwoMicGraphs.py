from enviro_calc import humid, sos
import matplotlib.pyplot as plt
from enviro_calc import vector2value
import numpy as np

# Set up 2 microphone array as static
H = np.array([[20, 0, 0], [40, 0, 0]], dtype=float)
number_points = 200
range_points = 60

def distance_multimic(a, H):
    """

    :param a: n x 3 array
    :param H: i x 3 array
    :return: n x i array of distance values
    """
    out = np.zeros((a.shape[0], H.shape[0]))
    for i in range(H.shape[0]):
        out[:, i] = np.sqrt(np.sum(np.square(H[i, :] - a), axis=1))
    return out


def error_2D_1D(height, temp_deg_assumed=20, rel_humidity=85, a=np.concatenate((np.linspace(0, range_points, number_points).reshape(-1, 1),np.zeros((number_points,1)),np.zeros((number_points,1))), axis=1)):
    """
    Calculates and plots the error in ToA to a microphone as a function of vertical height error, z
    :param v_sound:
    :param z:
    :return:
    """
    v_sound_assumed = sos(temp_deg_assumed, rel_humidity)

    # Set new z value for the sound source locatins, b
    height = height * np.ones((a.shape[0],1))
    b = np.concatenate((np.delete(a, -1, axis=1), height), axis=1)

    # Calculate TDOA, as this forms the basis of the error
    TDoA_3D = np.subtract(distance_multimic(b, H)[:, 0], distance_multimic(b, H)[:, 1]) / v_sound_assumed

    # Obtain the predicted differential distance
    diff_distance_predicted = TDoA_3D * v_sound_assumed

    # Obtain the actual differential distance (in 2D plane) by considering the difference in distance of the HORIZONTAL aspects of a rather than b
    diff_distance_actual = np.subtract(distance_multimic(a, H)[:, 0], distance_multimic(a, H)[:, 1])

    diff_distance_error = diff_distance_actual - diff_distance_predicted

    return TDoA_3D, diff_distance_error

def error_SoS_1D(dc, v_sound_assumed, a=np.concatenate((np.linspace(0, range_points, number_points).reshape(-1, 1),np.zeros((number_points,1)),np.zeros((number_points,1))), axis=1), wind=False):
    """
    TDoA: 1 x d where d is the no_points in a
    :param dc:
    :param v_sound_assumed:
    :return:
    """
    # Time difference of arrival between the first and second microphone using the ACTUAL speed of sound

    # If we are calculating wind, then dc is different for each sound source position a. The dc wind vector we pass in is the same shape as the returned distance array
    # So we divide each paths distance by the corresponding speed of sound to get the time, then subtract the two times to get TDoA
    if wind:
        TDoA = np.subtract(distance_multimic(a, H)[:, 0]/(v_sound_assumed + dc[:, 0]), distance_multimic(a, H)[:, 1]/(v_sound_assumed + dc[:, 1]))
        diff_distance_actual = np.subtract(distance_multimic(a, H)[:, 0], distance_multimic(a, H)[:, 1])
    else:
        TDoA = np.subtract(distance_multimic(a, H)[:, 0], distance_multimic(a, H)[:, 1]) / (v_sound_assumed + dc)
        diff_distance_actual = TDoA * (v_sound_assumed + dc)

    diff_distance_predicted = TDoA * v_sound_assumed

    diff_distance_error = diff_distance_actual - diff_distance_predicted

    return TDoA, diff_distance_error

def plot_variable_1D(variable_list, parameter='temperature'):
    variable_dict = {'temperature': error_temp_1D, 'wind': error_wind_1D, 'humidity': error_humidity_1D, '2D error': error_2D_1D}
    label_dict = {'temperature': "\u00b0C", 'wind': "m/s", 'humidity': "% RH", '2D error': 'm'}

    x = np.linspace(0, range_points, number_points).reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for item in variable_list:
        TDoA, Distance_error = variable_dict[parameter](item)
        ax1.plot(x, TDoA, label="%s %s" % (item, label_dict[parameter]))
        ax2.plot(x, Distance_error, label="%s %s" % (item, label_dict[parameter]))


    ax1.set_xlabel("x (m)", fontsize=16)
    ax1.set_ylabel("TDoA (s)", fontsize=16)
    ax2.set_ylabel("error (m)", fontsize=16)

    # Plot on microphones
    ax1.plot(H[0,0],H[0,1], 'ro', label="Microphone 1")
    ax1.plot(H[1,0],H[1,1],'rx', label='Microphone 2')
    ax2.plot(H[0, 0], H[0, 1], 'ro')
    ax2.plot(H[1, 0], H[1, 1], 'rx')

    fig.suptitle("%s for 20\u00b0C and 85%s RH assumed" % (parameter, '%'))

    ax1.legend()

    plt.show()

def plot_variable_2D(variable, parameter='temperature'):
    variable_dict = {'temperature': error_temp_1D, 'wind': error_wind_2D, 'humidity': error_humidity_1D, '2D error': error_2D_1D}
    label_dict = {'temperature': "\u00b0C", 'wind': "m/s", 'humidity': "% RH", '2D error': 'm'}

    x = np.linspace(0, range_points, number_points)
    y = np.linspace(-20, 20, number_points)

    # Set up a 2D meshgrid with x and y points
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Create an array of the sound source positions, made into a column vector for calculations
    a_2D = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), np.zeros_like(X.reshape(-1, 1))), axis=1)

    # Create a figure with two sub plots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Since this will be a contour plot, only a single variable at a time will be input to create a graph

    # Return the contour values from the error functions
    TDoA, Distance_error = variable_dict[parameter](variable, a=a_2D)

    # Reshape to the original meshgrid size (number_points x number_points)
    TDoA = TDoA.reshape(number_points, -1)
    Distance_error = Distance_error.reshape(number_points, -1)

    contours_ax1 = ax1.contourf(X, Y, TDoA)
    contours_ax2 = ax2.contourf(X, Y, Distance_error)

    cbar_ax1 = fig.colorbar(contours_ax1,ax=ax1)
    cbar_ax2 = fig.colorbar(contours_ax2, ax=ax2)

    #ax1.clabel(contours_ax1, inline=True, fontsize=16, fmt='%1.6f')
    #ax2.clabel(contours_ax2, inline=True, fontsize=16, fmt='%1.6f')

    ax1.set_xlabel("x (m)", fontsize=16)
    ax1.set_ylabel("y (m)", fontsize=16)

    # Plot on microphones
    ax1.plot(H[0,0],H[0,1], 'ro')#, label="Microphone 1")
    ax1.plot(H[1,0],H[1,1],'rx')#, label='Microphone 2')
    ax2.plot(H[0, 0], H[0, 1], 'ro')
    ax2.plot(H[1, 0], H[1, 1], 'rx')

    fig.suptitle("%s %s %s for 20\u00b0C and 85%s RH assumed" % (str(variable), label_dict[parameter], parameter, '%'))
    ax1.set_title("TDOA")
    ax2.set_title("Error (m)")

    plt.show()

def error_temp_1D(temp_deg, temp_deg_assumed=20, rel_humidity=85, a=np.concatenate((np.linspace(0, range_points, number_points).reshape(-1, 1),np.zeros((number_points,1)),np.zeros((number_points,1))), axis=1)):
    v_sound_actual = sos(temp_deg, rel_humidity)

    v_sound_assumed = sos(temp_deg_assumed, rel_humidity)

    dc = v_sound_actual - v_sound_assumed

    return error_SoS_1D(dc, v_sound_assumed, a=a)

def error_humidity_1D(rel_humidity, temp_deg_assumed=20, rel_humidity_assumed=85, a=np.concatenate((np.linspace(0, range_points, number_points).reshape(-1, 1),np.zeros((number_points,1)),np.zeros((number_points,1))), axis=1)):

    v_sound_actual = sos(temp_deg_assumed, rel_humidity)

    v_sound_assumed = sos(temp_deg_assumed, rel_humidity_assumed)

    dc = v_sound_actual - v_sound_assumed

    return error_SoS_1D(dc, v_sound_assumed, a=a)


def error_wind_1D(wind, temp_deg_assume=20, rel_humidity=85, a=np.concatenate((np.linspace(0, range_points, number_points).reshape(-1, 1),np.zeros((number_points,1)),np.zeros((number_points,1))), axis=1)):

    v_sound_assumed = sos(temp_deg_assume, rel_humidity)

    return error_SoS_1D(wind, v_sound_assumed, a=a)

def error_wind_2D(wind_vector, a, temp_deg_assume=20, rel_humidity=85):

    v_sound_assumed = sos(temp_deg_assume, rel_humidity)

    wind = vector2value(wind_vector, a, H)

    print(wind)

    return error_SoS_1D(wind, v_sound_assumed, a=a, wind=True)

temp = [10., 15., 20., 25., 30.]
wind = [-5., -2. - 1., 1., 2., 5., 10.]
humidity = [0., 20., 50., 85., 98.]
z = [0., 1., 2., 5.]

# plot_variable_1D(humidity, parameter='humidity')
# plot_variable_1D(temp, parameter='temperature')
# plot_variable_1D(wind, parameter='wind')
#plot_variable_1D(z, parameter='2D error')

wind_vector = np.array([2, 3, 0])

# plot_variable_2D(wind_vector, parameter='wind')
# plot_variable_2D(30, parameter='temperature')
# plot_variable_2D(20, parameter='humidity')
plot_variable_2D(5, parameter='2D error')