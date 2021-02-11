import numpy as np

from error_model import R, E_sos
from environ_param import humid, sos

import matplotlib.pyplot as plt

def plot_wind(xpoints, ypoints, H, wind, sensor_no):
    Z, indicator = calculate_error_contours(xpoints, ypoints, H, err_function='wind', wind=wind, sensor_no=sensor_no)

    contour_plot(xpoints, ypoints, Z, H)


def plot_temp(H, temp_deg, xpoints, ypoints, step, temp_assume=20):
    temp = temp_deg + 273.15  # Convert temperature into Kelvin
    temp_assume = temp_assume + 273.15

    dc = np.sqrt(1.402 * 8310 * temp / 28.966) - np.sqrt(
        1.402 * 8310 * temp_assume / 28.966)  # Get the difference in speed of sound from assumed

    v_sound = sos(temp_deg, 85)

    Z, indicator = calculate_error_contours(xpoints, ypoints, step, H, v_sound, err_function='dc', dc=dc)

    contour_plot(xpoints, ypoints, Z, H)


def plot_humidity(H, rel_humid, a_xpoints, a_ypoints, temp_assume=20):
    temp = temp_assume + 273.15  # Convert temperature into Kelvin

    abs_humidity, density = humid(temp_assume, rel_humid)

    spec_heat_ratio = (7 + abs_humidity) / (
            5 + abs_humidity)  # Specific heat ratio is dependent on the abs humidity, it is 1.4 at 0 humidity

    molar_mass = 28.966 * (
            1 - abs_humidity) + abs_humidity * 18.01528  # Get the new molar mass with that fraction of water

    dc = np.sqrt(spec_heat_ratio * 8310 * temp / molar_mass) - np.sqrt(
        1.402 * 8310 * temp / 28.966)  # Error in speed of sound estimate due to HUMIDITY only, so take the actual speed of sound with this temperature value

    Z, indicator = calculate_error_contours(xpoints, ypoints, H, err_function='dc', dc=dc)

    contour_plot(xpoints, ypoints, Z, H)


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


def plot_SNR(coeff, db_of_sound_source, dBthreshold=10):
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
    y = attenuation_eq(15 / 100 / 10 / np.log10(np.e), db_of_sound_source, x)
    plt.plot(x, y, color='brown', label='No wind open field')
    plt.title("SNR against distance from source at 80dB, 10kHz and 20°C", fontsize=20)
    plt.ylabel("SNR (dB)", fontsize=20)
    plt.xlabel("Distance from source (m)", fontsize=20)
    plt.legend(fontsize=26)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.show()


def plot_2D_nominal(xpoints, ypoints, H, v_sound, sensor_pair=(2, 4)):
    pass


def plot_sos_time_total(H, DT, dc, v_sound, a_xpoints, a_ypoints, a, increment=1):
    a_xmin = 5 - a_xpoints / 2
    a_ymin = 5 - a_ypoints / 2

    i_loop = int(a_xpoints / increment)
    j_loop = int(a_ypoints / increment)

    absolute_error_sos = np.zeros((i_loop, j_loop))
    absolute_error_time = np.zeros((i_loop, j_loop))

    x = np.arange(a_xmin, a_xpoints + a_xmin, increment)
    y = np.arange(a_ymin, a_ypoints + a_ymin, increment)

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

    contours = plt.contour(X, Y, absolute_error_sos)  # levels=[0.2,0.4, 0.6, 1], colors=['b', 'r', 'g'])
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


def calculate_error_contours(xpoints, ypoints, step, H, v_sound, err_function='dc', dc=0., a=np.array([0, 0, 0], dtype=float), DT=np.zeros((4, 1)),
                             sensor_no=0, wind=np.zeros((3, 1))):

    xmin = H[1][0] / 2 - xpoints / 2
    ymin = H[2][1] / 2 - ypoints / 2

    Z = np.zeros((ypoints, xpoints))

    a[2] = 0.01

    for i in range(xpoints):
        for j in range(ypoints):
            a[0] = xmin + step*i
            a[1] = ymin + step*j

            if err_function == 'dc':
                if v_sound == 0.:
                    print("\nInput speed of sound is 0. Please specify a speed of sound.")
                    SoS = float(input())

                    return calculate_error_contours(xpoints, ypoints, step, H, err_function=err_function, dc=dc, a=a, DT=DT,
                             sensor_no=sensor_no, wind=wind,v_sound=SoS)

                error_matrix = E_sos(a, H, dc, v_sound)

                Z[j][i] = np.sqrt(np.sum(np.square(error_matrix[:-1])))

                if Z[j][i] == 0:
                    Z[j][i] = Z[j-1]

            elif err_function == 'wind':
                line2sensor = H - a

                line2sensor = line2sensor / np.linalg.norm(
                    line2sensor)  # Normalize the line from sound source to sensor

                dc = np.dot(wind,
                            line2sensor)  # Dot product of wind and the line2 sensor gives the change in speed of sound

                error_matrix = E_sos(a, H,
                                     dc, v_sound)  # Calculate a particular error matrix for that speed of sound, will have to then have an error matrix FOR EACH MICROPHONE

                Z[j][i] = np.sqrt(np.sum(np.square(error_matrix[:-1])))

            elif err_function == '2D':
                print("\nIn this case the returned array is an array of arrays, with each element of the array being an array of the distance errors between pairs of microphones. \nEach of these sub-arrays SHOULD have 0 along their diagonals, and be (negatively/anti) symmetric")
                if v_sound == 0.:
                    print("\nInput speed of sound is 0. Please specify a speed of sound.")
                    SoS = float(input())

                    return calculate_error_contours(xpoints, ypoints, step, H, err_function=err_function, dc=dc, a=a, DT=DT,
                             sensor_no=sensor_no, wind=wind,v_sound=SoS)

                ToA_actual = np.divide(R(a, H, model-"3D"), v_sound)

                Dt_actual = ToA_actual - ToA_actual.T      # Could make this absolute value to get size of error

                diff = H[:][:-1] - a[:-1]

                h = np.sqrt(np.sum(np.dot(diff, diff.T)))

                ToA_predicted = h / v_sound

                Dt_predicted = ToA_predicted - ToA_predicted.T

                Dt = Dt_actual - Dt_predicted

                Z[j][i] = np.multiply(Dt, v_sound)  # To get the time error into a distance error. This means Z will return as an array of arrays for this particular case

            elif err_function == 'SNR':
                pass

            else:
                pass

    indicator = calculate_error_indicator(Z)

    return Z, indicator


def contour_plot(xpoints, ypoints, Z, H, title=None):
    plt.figure(1)

    a_xmin = 5 - xpoints / 2
    a_ymin = 5 - ypoints / 2

    x = np.arange(a_xmin, xpoints + a_xmin, 1)
    y = np.arange(a_ymin, ypoints + a_ymin, 1)

    X, Y = np.meshgrid(x, y, indexing = 'xy')

    contours = plt.contour(X, Y, Z)
    plt.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')

    plt.plot(H[0][0], H[0][1], 'o', color='black', label="Microphone positions")
    plt.plot(H[1][0], H[1][1], 'o', color='black')
    plt.plot(H[2][0], H[2][1], 'o', color='black')
    plt.plot(H[3][0], H[3][1], 'o', color='black')
    plt.legend(framealpha=1)

    if title:
        plt.title("%s" % title, wrap=True,
                  fontsize=16)

    plt.xlabel("x (m)", fontsize=16)
    plt.ylabel("y (m)", fontsize=16)

    plt.show()


h1 = [0, 0, 0]
h2 = [0, 10, 0]
h3 = [10, 10, 0]
h4 = [10, 0, 0]
H = np.array([h1, h2, h3, h4], dtype=float)


xpoints = 100
ypoints = 100
x = np.linspace(-20, 20, xpoints)
step = 40/100

v_sound = sos(20,85)

plot_temp(H, 20, xpoints, ypoints, step)
