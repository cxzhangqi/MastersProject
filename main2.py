import numpy as np
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

def R(i, a_x, a_y, a_z, H):
    """
    Calculates the distance from source to microphone i.
    :param i: Microphone
    :param a_x: x-
    :param a_y:
    :param a_z:
    :param H:
    :return:
    """
    return np.sqrt(np.power((H[i, 0] - a_x), 2) + np.power((H[i, 1] - a_y), 2) + np.power((H[i, 2] - a_z), 2))


def m(i, j, a_x, a_y, a_z, H):
    """
    Returns the value of M[i,j] with sound source, a, and microphone positions H.

    """
    if j == 0:
        choice = a_x
    elif j == 1:
        choice = a_y
    else:
        choice = a_z
    if R(i, a_x, a_y, a_z, H) == 0:
        return 0
    else:
        return np.divide(np.subtract(H[i][j], choice), R(i, a_x, a_y, a_z, H))


def d(i, j, a_x, a_y, a_z, H, v_sound):
    if j == 0:
        choice = a_x
    elif j == 1:
        choice = a_y
    else:
        choice = a_z

    if R(i, a_x, a_y, a_z, H) == 0:
        return 0
    else:
        return np.divide(np.subtract(H[i][j], choice), np.multiply(v_sound, R(i, a_x, a_y, a_z, H)))


def E_sos(a_x, a_y, a_z, H, dc):
    """
    Returns a 4x1 matrix containing the position estimate errors for error in speed of sound, dc
    First index is error in x
    Second index is error iny
    Third index is...
    :param a:
    :param H:
    :return:
    """
    M = np.zeros((4, 4))

    for i in range(4):
        M[i][3] = 1
        for j in range(3):
            M[i][j] = m(i, j, a_x, a_y, a_z, H)

    if np.linalg.det(M) == 0:  # Need to figure out a better way of handling this. For now just do this?
        print("\n  SOS buggered")
    M_inv = np.linalg.inv(M)

    T = np.array([[np.divide(R(0, a_x, a_y, a_z, H), v_sound)], [np.divide(R(1, a_x, a_y, a_z, H), v_sound)],
                  [np.divide(R(2, a_x, a_y, a_z, H), v_sound)], [np.divide(R(3, a_x, a_y, a_z, H), v_sound)]])

    return np.multiply(np.matmul(M_inv, T), dc)  # This is a 4x1 array


def E_time(a_x, a_y, a_z, H, DT, v_sound):
    D = np.zeros((4, 4))

    for i in range(4):
        D[i][3] = 1
        for j in range(3):
            D[i][j] = d(i, j, a_x, a_y, a_z, H, v_sound)

    if np.linalg.det(D) == 0:  # Need to figure out a better way of handling this. For now just do this?
        print("\n We got here!")

    D_inv = np.linalg.inv(D)

    return np.matmul(D_inv, DT)  # Should be a 3 x 1 array


def dDt_dv_ij(i, j, a_x, a_y, a_z, H, v_sound):
    h_i = np.sqrt(np.power((a_x - H[i][0]), 2) + np.power((a_y - H[i][1]), 2))

    h_j = np.sqrt(np.power((a_x - H[j][0]), 2) + np.power((a_y - H[j][1]), 2))

    return np.multiply(np.divide(a_z, v_sound), np.subtract(np.divide(1, np.sqrt(np.power(a_z, 2) + np.power(h_i, 2))),
                                                            np.divide(1, np.sqrt(np.power(a_z, 2) + np.power(h_j, 2)))))


# plot hyperbola(a,)
# For now, just do it between two microphones rather than getting into a matrix.
def E_2D(a_x, a_y, a_z, H, v_sound, i, j):
    dDt_dv = dDt_dv_ij(i, j, a_x, a_y, a_z, H, v_sound)

    return np.multiply(a_z, dDt_dv)  # Returns the change in arrival time between microphones i and j


# I think this may be wrong but it looks good
def plot_2D_nominal(H, v_sound, a_xrange, a_yrange, a_z=1, increment=1):
    a_xmin = 5 - a_xrange / 2
    a_ymin = 5 - a_yrange / 2

    i_loop = int(a_xrange / increment)
    j_loop = int(a_yrange / increment)

    nominal_error_2D_microphone2_4 = np.zeros((i_loop, j_loop))
    nominal_error_2D_microphone4_2 = np.zeros((i_loop, j_loop))
    nominal_error_2D_microphone3 = np.zeros((i_loop, j_loop))
    nominal_error_2D_microphone4 = np.zeros((i_loop, j_loop))

    x = np.arange(a_xmin, a_xrange + a_xmin, increment)
    y = np.arange(a_ymin, a_yrange + a_ymin, increment)

    for i in range(i_loop):
        for j in range(j_loop):
            a_x = a_xmin + i
            a_y = a_ymin + j
            a = [a_x, a_y, a_z]

            h_i = np.sqrt(np.power((a_x - H[1][0]), 2) + np.power((a_y - H[1][1]), 2))

            h_j = np.sqrt(np.power((a_x - H[3][0]), 2) + np.power((a_y - H[3][1]),
                                                                  2))  # Change these two into a function to get any difference in TDOA

            Dt_actual = np.divide(
                np.subtract(np.sqrt(np.power(a_z, 2) + np.power(h_i, 2)), np.sqrt(np.power(a_z, 2) + np.power(h_j, 2))),
                v_sound)

            Dt_predicted = np.divide(np.subtract(h_i, h_j), v_sound)

            Dt = Dt_actual - Dt_predicted

            nominal_error_2D_microphone2_4[j][i] = np.multiply(Dt,
                                                               v_sound)  # To get the time error into a distance error

            Dt_actual = np.divide(
                np.subtract(np.sqrt(np.power(a_z, 2) + np.power(h_j, 2)), np.sqrt(np.power(a_z, 2) + np.power(h_i, 2))),
                v_sound)

            Dt_predicted = np.divide(np.subtract(h_j, h_i), v_sound)

            Dt = Dt_actual - Dt_predicted

            nominal_error_2D_microphone4_2[j][i] = np.multiply(Dt, v_sound)

    positive_error = np.absolute(nominal_error_2D_microphone2_4)

    indicator = calculate_error_indicator(positive_error)

    X, Y = np.meshgrid(x, y, indexing='xy')

    plt.figure(1)  # Figure

    contours = plt.contour(X, Y, nominal_error_2D_microphone2_4)
    plt.clabel(contours, inline=True, fontsize=16,fmt='%1.1f')
    # plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='red', label="Microphone 2")
    # plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    # plt.text(4, 5, "<0.4m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend(framealpha=1)
    plt.title("Nominal error for microphone 2 and 4 with a sound source height of %0.1f m"%(a_z),wrap=True,fontsize=16)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator),fontsize=16)
    plt.ylabel("y (m)",fontsize=16)
    # plt.figure(2)  # Figure
    #
    # contours = plt.contour(X, Y, nominal_error_2D_microphone4_2)
    # plt.clabel(contours, inline=True, fontsize=10)
    # # plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    # plt.plot(10, 0, 'o', color='black')
    # # plt.plot(10, 10, 'o', color='black')
    # plt.plot(0, 10, 'o', color='red', label="Microphone 4")
    # # plt.text(4, 5, "<0.4m")
    # # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    # plt.legend()
    # plt.title("Nominal")

    plt.show()


def plot_sos_time_total(H, DT, dc, v_sound, a_xrange, a_yrange, a_z=0.001, increment=1):
    a_xmin = 5 - a_xrange / 2
    a_ymin = 5 - a_yrange / 2

    i_loop = int(a_xrange / increment)
    j_loop = int(a_yrange / increment)

    absolute_error_sos = np.zeros((i_loop, j_loop))
    absolute_error_time = np.zeros((i_loop, j_loop))

    x = np.arange(a_xmin, a_xrange + a_xmin, increment)
    y = np.arange(a_ymin, a_yrange + a_ymin, increment)

    for i in range(i_loop):
        for j in range(j_loop):
            a_x = a_xmin + i
            a_y = a_ymin + j
            a = [a_x, a_y, a_z]

            error_matrix_sos = E_sos(a_x, a_y, a_z, H, dc)
            error_matrix_time = E_time(a_x, a_y, a_z, H, DT, v_sound)

            # if (error_matrix_sos[0] == 10000) & (error_matrix_time[0] == 10000):
            #     absolute_error_sos[j][i] = absolute_error_sos[j][i-1]
            #     absolute_error_time[j][i] = absolute_error_time[j][i - 1]
            # elif error_matrix_sos[0] == 10000:
            #     absolute_error_sos[j][i] = absolute_error_sos[j][i - 1]
            # elif error_matrix_time[0] == 10000:
            #     absolute_error_time[j][i] = absolute_error_time[j][i-1]
            # else:
            absolute_error_sos[j][i] = np.sqrt(
                np.power(error_matrix_sos[0][0], 2) + np.power(error_matrix_sos[1][0], 2) + np.power(
                    error_matrix_sos[3][0],
                    2))
            absolute_error_time[j][i] = np.sqrt(
                np.power(error_matrix_time[0][0], 2) + np.power(error_matrix_time[1][0], 2) + np.power(
                    error_matrix_time[3][0], 2))

    absolute_error = absolute_error_sos + absolute_error_time

    indicator = calculate_error_indicator(absolute_error)

    print("Indicative error: %f" % (indicator))

    X, Y = np.meshgrid(x, y, indexing='xy')

    plt.figure(1)

    # Z = np.sqrt(np.power(E_vec(X, Y, H, dc)[0][0], 2) + np.power(E_vec(X, Y, H, dc)[1][0], 2))

    contours = plt.contour(X, Y, absolute_error_sos, )  # levels=[0.2,0.4, 0.6, 1], colors=['b', 'r', 'g'])
    plt.clabel(contours, inline=True, fontsize=10)
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.4m")
    plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend()

    plt.figure(2)

    # Z = np.sqrt(np.power(E_vec(X, Y, H, dc)[0][0], 2) + np.power(E_vec(X, Y, H, dc)[1][0], 2))

    contours = plt.contourf(X, Y, absolute_error_time)
    # plt.clabel(contours, inline=True, fontsize=10)
    plt.colorbar()
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.4m")
    plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend()

    plt.figure(3)

    # Z = np.sqrt(np.power(E_vec(X, Y, H, dc)[0][0], 2) + np.power(E_vec(X, Y, H, dc)[1][0], 2))

    contours = plt.contour(X, Y, absolute_error)
    plt.clabel(contours, inline=True, fontsize=10)
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.4m")
    plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend()

    # plt.close('all')
    plt.show()


# Trying to do wind direction
# We want to change the speed of sound, depending on which sensor the sound is travelling to AS WELL AS the position of the sound source.
# So would want a loop, for each position a then you get a new error in speed of sound, for each sensor you get an error. Nice

def plot_wind_mic2_4(H, dir_wind, v_wind, a_xrange, a_yrange, a_z=0.001, increment=1):
    a_xmin = 5 - a_xrange / 2
    a_ymin = 5 - a_yrange / 2

    i_loop = int(a_xrange / increment)
    j_loop = int(a_yrange / increment)

    x = np.arange(a_xmin, a_xrange + a_xmin, increment)
    y = np.arange(a_ymin, a_yrange + a_ymin, increment)

    absolute_error_wind_microphone1 = np.zeros((i_loop, j_loop))
    absolute_error_wind_microphone2 = np.zeros((i_loop, j_loop))
    absolute_error_wind_microphone3 = np.zeros((i_loop, j_loop))
    absolute_error_wind_microphone4 = np.zeros((i_loop, j_loop))

    for i in range(i_loop):
        for j in range(j_loop):
            a_x = a_xmin + i
            a_y = a_ymin + j
            a_z = 0.00001
            a = np.array([a_x, a_y, a_z])

            for k in range(4):
                line_to_sensor = np.subtract(H[k], a)

                line_to_sensor = line_to_sensor / np.linalg.norm(
                    line_to_sensor)  # Normalize the line from sound source to sensor

                dc = np.multiply(np.dot(dir_wind, line_to_sensor),
                                 v_wind)  # Multiply the direction coefficient (dot product) by wind speed to get the change in speed of sound for the particular sensor
                print(dc)
                error_matrix_sos = E_sos(a_x, a_y, a_z, H,
                                         dc)  # Calculate a particular error matrix for that speed of sound, will have to then have an error matrix FOR EACH MICROPHONE

                if k == 0:
                    absolute_error_wind_microphone1[j][i] = np.sqrt(
                        np.power(error_matrix_sos[0][0], 2) + np.power(error_matrix_sos[1][0], 2) + np.power(
                            error_matrix_sos[3][0],
                            2))

                elif k == 1:
                    absolute_error_wind_microphone2[j][i] = np.sqrt(
                        np.power(error_matrix_sos[0][0], 2) + np.power(error_matrix_sos[1][0], 2) + np.power(
                            error_matrix_sos[3][0],
                            2))

                elif k == 2:
                    absolute_error_wind_microphone3[j][i] = np.sqrt(
                        np.power(error_matrix_sos[0][0], 2) + np.power(error_matrix_sos[1][0], 2) + np.power(
                            error_matrix_sos[3][0],
                            2))

                else:
                    absolute_error_wind_microphone4[j][i] = np.sqrt(
                        np.power(error_matrix_sos[0][0], 2) + np.power(error_matrix_sos[1][0], 2) + np.power(
                            error_matrix_sos[3][0],
                            2))

    indicator = calculate_error_indicator(absolute_error_wind_microphone2)

    print("Indicative error for microphone 2 with wind speed of %d and direction vector [1,1,0]: %f" % (
        v_wind, indicator))

    X, Y = np.meshgrid(x, y, indexing='xy')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    contours = ax1.contour(X, Y, absolute_error_wind_microphone2)
    ax1.clabel(contours, inline=True, fontsize=16,fmt='%1.1f')
    ax1.plot(0, 0, 'o', color='black', label="Microphone positions")
    ax1.plot(10, 0, 'o', color='green', label="Microphone 2")
    ax1.plot(10, 10, 'o', color='black')
    ax1.plot(0, 10, 'o', color='red')
    ax1.text(4, 5, "<0.2m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    fig.suptitle("Absolute position error for microphone 2 and 4 \nwith wind speed %0.1f m/s and direction 45°" % (v_wind),wrap=True,fontsize=20)
    ax1.set_xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator),fontsize=20)
    ax1.set_ylabel("y (m)",fontsize=20)

    indicator = calculate_error_indicator(absolute_error_wind_microphone4)

    contours = ax2.contour(X, Y, absolute_error_wind_microphone4)
    ax2.clabel(contours, inline=True, fontsize=16,fmt='%1.1f')
    ax2.plot(0, 0, 'o', color='black', label="Microphones")
    ax2.plot(10, 0, 'o', color='green',label="Microphone 2")
    ax2.plot(10, 10, 'o', color='black')
    ax2.plot(0, 10, 'o', color='red', label='Microphone 4')
    ax2.text(4, 5, "<0.2m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    ax2.legend(framealpha=1,fontsize=14)
    #ax2.title("Absolute position error for microphone 4 wind speed %0.1f and direction 45°" % (v_wind),wrap=True)
    ax2.set_xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator),fontsize=20)
    # ax2.ylabel("y (m)")

    plt.show()


def sos(temp_deg, rel_humid):
    temp_k = temp_deg + 273.15

    abs_humidity, density = humid(temp_deg, rel_humid)

    spec_heat_ratio = (7 + abs_humidity) / (
            5 + abs_humidity)  # Specific heat ratio is dependent on the abs humidity, it is 1.4 at 0 humidity

    molar_mass = 28.966 * (
            1 - abs_humidity) + abs_humidity * 18.01528  # Get the new molar mass with that fraction of water

    sos = np.sqrt(spec_heat_ratio * 8310 * temp_k / molar_mass)

    return sos


def plot_temp(H, temp_deg, a_xrange, a_yrange, temp_assume=20, a_z=0.001, increment=1):
    temp = temp_deg + 273.15  # Convert temperature into Kelvin
    temp_assume = temp_assume + 273.15
    dt = temp - temp_assume

    a_xmin = 5 - a_xrange / 2
    a_ymin = 5 - a_yrange / 2

    i_loop = int(a_xrange / increment)
    j_loop = int(a_yrange / increment)

    absolute_error_temp = np.zeros((i_loop, j_loop))

    x = np.arange(a_xmin, a_xrange + a_xmin, increment)
    y = np.arange(a_ymin, a_yrange + a_ymin, increment)

    dc = np.sqrt(1.402 * 8310 * temp / 28.966) - np.sqrt(
        1.402 * 8310 * temp_assume / 28.966)  # Get the difference in speed of sound from assumed

    print(dc)

    for i in range(i_loop):
        for j in range(j_loop):
            a_x = a_xmin + i
            a_y = a_ymin + j
            a = [a_x, a_y, a_z]

            error_matrix_temp = E_sos(a_x, a_y, a_z, H, dc)

            absolute_error_temp[j][i] = np.sqrt(
                np.power(error_matrix_temp[0][0], 2) + np.power(error_matrix_temp[1][0], 2) + np.power(
                    error_matrix_temp[3][0],
                    2))

    indicator = calculate_error_indicator(absolute_error_temp)

    print("Indicative error for temperature deviation of %d: %f" % (dt, indicator))

    X, Y = np.meshgrid(x, y, indexing='xy')

    plt.figure(1)

    # Z = np.sqrt(np.power(E_vec(X, Y, H, dc)[0][0], 2) + np.power(E_vec(X, Y, H, dc)[1][0], 2))

    contours = plt.contour(X, Y, absolute_error_temp)  # levels=[0.2,0.4, 0.6, 1], colors=['b', 'r', 'g'])
    plt.clabel(contours, inline=True, fontsize=16,fmt='%1.1f')
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.25m")
    #plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend(framealpha=1)
    plt.title("Absolute position error for temperature error $\Delta$%0.1f°C" % (dt),fontsize=20, wrap=True)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator),fontsize=20)
    plt.ylabel("y (m)",fontsize=20)
    plt.show()


def humid(temp_deg, rel_humid):
    vapor_pressure = 1000 * 0.61121 * np.exp((18.678 - temp_deg / 234.5) * (
            temp_deg / (257.14 + temp_deg)))  # Approximate vapor_pressure with the Buck equation

    p_v = rel_humid / 100 * vapor_pressure

    p_d = 1.013e5 - p_v

    density_humid = (p_d * 0.028966 + p_v * 0.018016) / 8.314 / (temp_deg + 273.15)

    abs_humidity = 0.01 * rel_humid * vapor_pressure / 1.013e5  # Absolute humidity value based on temperature (for vapor pressure) and relative humidity

    return abs_humidity, density_humid

def plot_humid(H, temp_deg, rel_humid, a_xrange, a_yrange, temp_assume=20, a_z=0.001, increment=1):
    temp = temp_deg + 273.15  # Convert temperature into Kelvin

    abs_humidity, density = humid(temp_deg, rel_humid)

    spec_heat_ratio = (7 + abs_humidity) / (
            5 + abs_humidity)  # Specific heat ratio is dependent on the abs humidity, it is 1.4 at 0 humidity

    molar_mass = 28.966 * (
            1 - abs_humidity) + abs_humidity * 18.01528  # Get the new molar mass with that fraction of water

    dc = np.sqrt(spec_heat_ratio * 8310 * temp / molar_mass) - np.sqrt(
        1.402 * 8310 * temp / 28.966)  # Error in speed of sound estimate due to HUMIDITY only, so take the actual speed of sound with this temperature value

    a_xmin = 5 - a_xrange / 2
    a_ymin = 5 - a_yrange / 2

    i_loop = int(a_xrange / increment)
    j_loop = int(a_yrange / increment)

    absolute_error_humid = np.zeros((i_loop, j_loop))

    x = np.arange(a_xmin, a_xrange + a_xmin, increment)
    y = np.arange(a_ymin, a_yrange + a_ymin, increment)

    print(dc)

    for i in range(i_loop):
        for j in range(j_loop):
            a_x = a_xmin + i
            a_y = a_ymin + j
            a = [a_x, a_y, a_z]

            error_matrix_humid = E_sos(a_x, a_y, a_z, H, dc)

            absolute_error_humid[j][i] = np.sqrt(
                np.power(error_matrix_humid[0][0], 2) + np.power(error_matrix_humid[1][0], 2) + np.power(
                    error_matrix_humid[3][0],
                    2))

    indicator = calculate_error_indicator(absolute_error_humid)

    X, Y = np.meshgrid(x, y, indexing='xy')

    plt.figure(1)

    # Z = np.sqrt(np.power(E_vec(X, Y, H, dc)[0][0], 2) + np.power(E_vec(X, Y, H, dc)[1][0], 2))

    contours = plt.contour(X, Y, absolute_error_humid)  # levels=[0.2,0.4, 0.6, 1], colors=['b', 'r', 'g'])
    plt.clabel(contours, inline=True, fontsize=16,fmt='%1.1f')
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.08m")
    #plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend()
    plt.title("Absolute position error for humidity error %0.1f RH" % (rel_humid), fontsize=20)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator),fontsize=20)
    plt.ylabel("y (m)",fontsize=20)

    plt.show()


# Now we want to calculate some sort of error integral, to add up all the error in a box and then we can get some ranking of how important the parameter is.

def calculate_error_indicator(error_matrix):
    # Integrate
    sum = 0
    for i in range(5, 15, 1):
        for j in range(5, 15, 1):
            sum += error_matrix[j][i]

    # Return error square divided by area of the box
    return np.divide(np.power(sum, 2), 100)


def plot_pair_sourceHeight(h1, h2, a_x, a_z):
    plt.figure(1)

    h1 = np.array(h1)
    h2 = np.array(h2)
    line_length = np.linalg.norm(h2 - h1)

    # x = np.linspace(0,line_length+1,num=20)
    # z = np.linspace(0,a_z+2,num=20)
    # plt.plot(a_x,a_z)

    x_values = [0, line_length]
    y_values = [0, 0]
    plt.plot(x_values, y_values, '-', color='black')
    length = a_x
    plt.text(a_x / 2, 0.1, "%d" % length)

    x_values[1] = a_x
    y_values[1] = a_z
    plt.plot(x_values, y_values, '--', color='black')
    length = np.sqrt(np.power(a_x, 2) + np.power(a_z, 2))
    plt.text(a_x / 2, 1.4, "%0.2f" % length)

    x_values[0] = a_x
    y_values[0] = 0
    plt.plot(x_values, y_values, '--', color='black')
    length = a_z
    plt.text(a_x + 0.2, a_z / 2, "%d" % length)

    plt.plot(a_x, a_z, 'or')
    plt.xlim(-1, line_length + 1)
    plt.ylim(-1, line_length / 4)
    plt.plot(0, 0, 'o', color='green', label='Microphone 2')
    plt.plot(line_length, 0, 'o', color='green', label='Microphone 4')
    plt.legend()
    plt.show()


def attenuation_coeff(f, temp_deg, rel_humid):
    # Convert frequency in Hz to radians
    w = f * 2 * np.pi
    print(w)

    # Convert temp to Kelvin
    temp_k = temp_deg + 273.15

    # Density
    abs_humid, density = humid(temp_deg, rel_humid)
    print(density)

    # Speed of sound
    speed_of_sound = sos(temp_deg, rel_humid)
    print(speed_of_sound)

    # Dynamic viscosity
    dyn_visc = 1.825e-5 * np.power(temp_k / 293.15, 0.7)
    print(dyn_visc)
    # 0.001792 * np.power(np.e,(-1.94-4.8*273.16/temp_k+6.74*np.power(273.16/temp_k,2)))

    # 1.458e-6 * np.power(temp_k,1.5) / (temp_k + 110.4)

    coeff = 2 * dyn_visc * np.power(w, 2) / 3 / density / np.power(speed_of_sound, 3)

    return coeff  # Use the Stokes equation to calculate the attenuation coefficient


def attenuation_eq(coeff, A_0, x):

    #A = A_0/np.power(x,2)*np.power(np.e,-coeff*x/2)
    #
    # A = 10*np.log10(A_0/np.power(x,2)) - coeff*np.power(x,2)/2*10*np.log10(np.e)
    #
    # A = A_0 + A

    y = 20*np.log10(x) + A_0*(1 - np.power(np.e,-coeff*x/2))     # NEED TO CHECK IF THIS IS CORRECT

    # x_solve = sympy.symbols('x_solve')
    #
    # expr = A_0 - 20*sympy.log(x_solve,10) + A_0*(1 - np.power(np.e,-coeff*x_solve/2))
    #
    # zero_crossing = solve(A_0 - 20*np.log10(x_solve) + A_0*(1 - np.power(np.e,-coeff*x_solve/2)),x)

    return A_0 - y#, zero_crossing[0]


def plot_SNR(coeff,db_of_sound_source,dBthreshold=10):
    ymin = 0
    xmax = 50

    db_of_sound_source -= 30

    plt.figure(1)
    plt.xlim(0, xmax)
    plt.ylim(0, db_of_sound_source)
    x = np.arange(0, xmax, 1)
    y = attenuation_eq(coeff, db_of_sound_source, x)
    plt.plot(x, y, color='red', label='80% RH')

    # x_values = [0, xmax]
    # y_values = [dBthreshold, dBthreshold]
    # plt.plot(x_values, y_values, '--')

    y = attenuation_eq(0.065, db_of_sound_source, x)
    plt.plot(x, y, color='green', label='20% RH')

    y = attenuation_eq(coeff * sos(20, 0) / ((sos(20, 0)) - 20), db_of_sound_source, x)
    plt.plot(x, y, color='blue', label='Wind speed -20m/s')

    # y = attenuation_eq(0.065*sos(20,0)/((sos(20,0))-10),db_of_sound_source, x)
    # plt.plot(x, y, color='green',label='Attenuation with 20% RH')

    # y = attenuation_eq((30 / 100 / 10 / np.log10(np.e)), db_of_sound_source,
    #                    x)  # THese could be heaps wrong, but we'll see. Also check under what conditions they did each of these stats.
    # plt.plot(x, y, color='black', label='Deciduous forest')
    #
    y= attenuation_eq(15 / 100 / 10 / np.log10(np.e), db_of_sound_source, x)
    plt.plot(x, y, color='brown', label='No wind open field')
    plt.title("SNR against distance from source at 80dB, 10kHz and 20°C",fontsize=20)
    plt.ylabel("SNR (dB)",fontsize=20)
    plt.xlabel("Distance from source (m)",fontsize=20)
    plt.legend(fontsize=26)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.show()

#plot_sos_time_total(H,DT,dc,v_sound,a_xrange,a_yrange)

plot_wind_mic2_4(H,dir_wind,v_wind,a_xrange,a_yrange)

plot_temp(H,temp_deg,a_xrange,a_yrange)
#
# plot_humid(H,temp_deg,rel_humid,a_xrange,a_yrange)
#
# plot_2D_nominal(H, v_sound, a_xrange, a_yrange, a_z=2)
#
# plot_pair_sourceHeight(h2d,h4d,10,1)

#plot_SNR(coeff,db_of_sound_source)
