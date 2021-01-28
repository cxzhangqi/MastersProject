import numpy as np

import matplotlib as plt

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
    ax1.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')
    ax1.plot(0, 0, 'o', color='black', label="Microphone positions")
    ax1.plot(10, 0, 'o', color='green', label="Microphone 2")
    ax1.plot(10, 10, 'o', color='black')
    ax1.plot(0, 10, 'o', color='red')
    ax1.text(4, 5, "<0.2m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    fig.suptitle(
        "Absolute position error for microphone 2 and 4 \nwith wind speed %0.1f m/s and direction 45°" % (v_wind),
        wrap=True, fontsize=20)
    ax1.set_xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator), fontsize=20)
    ax1.set_ylabel("y (m)", fontsize=20)

    indicator = calculate_error_indicator(absolute_error_wind_microphone4)

    contours = ax2.contour(X, Y, absolute_error_wind_microphone4)
    ax2.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')
    ax2.plot(0, 0, 'o', color='black', label="Microphones")
    ax2.plot(10, 0, 'o', color='green', label="Microphone 2")
    ax2.plot(10, 10, 'o', color='black')
    ax2.plot(0, 10, 'o', color='red', label='Microphone 4')
    ax2.text(4, 5, "<0.2m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    ax2.legend(framealpha=1, fontsize=14)
    # ax2.title("Absolute position error for microphone 4 wind speed %0.1f and direction 45°" % (v_wind),wrap=True)
    ax2.set_xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator), fontsize=20)
    # ax2.ylabel("y (m)")

    plt.show()


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
    plt.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.25m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend(framealpha=1)
    plt.title("Absolute position error for temperature error $\Delta$%0.1f°C" % (dt), fontsize=20, wrap=True)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator), fontsize=20)
    plt.ylabel("y (m)", fontsize=20)
    plt.show()


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
    plt.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')
    plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='black')
    plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    plt.text(4, 5, "<0.08m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend()
    plt.title("Absolute position error for humidity error %0.1f RH" % (rel_humid), fontsize=20)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator), fontsize=20)
    plt.ylabel("y (m)", fontsize=20)

    plt.show()


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
    plt.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')
    # plt.plot(0, 0, 'o', color='black', label="Microphone positions")
    plt.plot(10, 0, 'o', color='red', label="Microphone 2")
    # plt.plot(10, 10, 'o', color='black')
    plt.plot(0, 10, 'o', color='black')
    # plt.text(4, 5, "<0.4m")
    # plt.gca().add_patch(plt.Polygon(H2D, facecolor=None, fill=False))
    plt.legend(framealpha=1)
    plt.title("Nominal error for microphone 2 and 4 with a sound source height of %0.1f m" % (a_z), wrap=True,
              fontsize=16)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator), fontsize=16)
    plt.ylabel("y (m)", fontsize=16)
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

def contour_plot(xrange, yrange, Z, H):
    plt.figure(1)

    a_xmin = 5 - xrange / 2
    a_ymin = 5 - yrange / 2

    x = np.arange(a_xmin, xrange + a_xmin, 1)
    y = np.arange(a_ymin, yrange + a_ymin, 1)

    contours = plt.contour(X, Y, Z)
    plt.clabel(contours, inline=True, fontsize=16, fmt='%1.1f')

    plt.plot(H[0][0], H[0][1], 'o', color='black', label="Microphone positions")
    plt.plot(H[1][0], H[1][1], 'o', color='red', label="Microphone 2")
    plt.plot(H[2][0], H[2][1], 'o', color='black')
    plt.plot(H[3][0], H[3][1], 'o', color='black')
    plt.legend(framealpha=1)
    plt.title("Nominal error for microphone 2 and 4 with a sound source height of %0.1f m" % (a_z), wrap=True,
              fontsize=16)
    plt.xlabel("x (m)\n\nIndicative error is %0.2f" % (indicator), fontsize=16)
    plt.ylabel("y (m)", fontsize=16)