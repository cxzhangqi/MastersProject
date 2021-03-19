import numpy as np
import matplotlib.pyplot as plt


def humid(temp_deg, rel_humid):
    """ Returns the absolute humidity as a fraction and a percentage.
    Third return is the density of air at this humidity."""
    vapor_pressure = 1000 * 0.61121 * np.exp((18.678 - temp_deg / 234.5) * (
            temp_deg / (257.14 + temp_deg)))  # Approximate vapor_pressure with the Buck equation

    p_v = rel_humid / 100 * vapor_pressure

    p_d = 1.013e5 - p_v

    density_humid = (p_d * 0.028966 + p_v * 0.018016) / 8.314 / (temp_deg + 273.15)

    abs_humidity_frac = 0.01 * rel_humid * vapor_pressure / 1.013e5  # Absolute humidity value based on temperature (for vapor pressure) and relative humidity

    abs_humidity_percent = abs_humidity_frac * 100

    return abs_humidity_frac, abs_humidity_percent, density_humid


def sos(temp_deg, RH, pressure=np.array([1.01325e5]), ppm_conc_co2=400):
    """Using the 1993 ISO standard it calcualtes the speed of sound in air given the molar fraction of CO_2, temperature
     in degrees, relative humidity and pressure. Inputs can be vectors"""

    # Vectorize all inputs
    temp_deg = np.array(temp_deg, dtype=np.float64)
    RH = np.array(RH, dtype=np.float64)
    pressure = np.array(pressure, dtype=np.float64)
    molar_conc_co2 = np.array(ppm_conc_co2, dtype=np.float64) / 1e6

    # First calculate the molar fraction of water vapor in air based on RH
    pass


def sos_old(temp_deg, rel_humid):
    """ Returns the speed of sound for a given temperature and relative humidity."""
    temp_k = temp_deg + 273.15

    abs_humidity_frac, abs_humidity_percent, density = humid(temp_deg, rel_humid)

    spec_heat_ratio = (7 + abs_humidity_frac) / (
            5 + abs_humidity_frac)  # Specific heat ratio is dependent on the abs humidity, it is 1.4 at 0 humidity

    molar_mass = 28.966 * (
            1 - abs_humidity_frac) + abs_humidity_frac * 18.01528  # Get the new molar mass with that fraction of water

    sos = np.sqrt(spec_heat_ratio * 8310 * temp_k / molar_mass)

    return sos


def saturation_pressure_water(temp_deg):
    """Returns the saturation concentration of water vapour at a specific tempreature using ISO 9613"""

    return (4.6151 - 6.8346 * (273.16 / (temp_deg + 273.15)) ** 1.261)


def molar_concentration_water(temp_deg, RH, pressure=1.01325e5):
    """Returns the absolute molar concentration of water vapour in air for a given temperature,
    pressure and relative humidity"""

    C = saturation_pressure_water(temp_deg)

    return RH * (10 ** C) / pressure * 1.01325e5


def pressure_at_x_in_Pa(initial_pressure, x, temp_deg=20, freq=5e3, RH=80, pressure=1.01325e5):
    """Returns the pressure (Pa) at a distance x given initial pressures, temperatures frequencies and RH"""

    initial_pressure = np.array(initial_pressure, dtype=np.float64)
    freq = np.array(freq, dtype=np.float64)
    temp_deg = np.array(temp_deg, dtype=np.float64)
    RH = np.array(RH, dtype=np.float64)
    pressure = np.array(pressure, dtype=np.float64)

    alpha = attenuation_coefficient_dBperm(freq, temp_deg, RH, pressure=pressure)

    scale_factor_from_dB = 1 / (20 * np.log10(np.e))

    return initial_pressure * np.exp(- x * alpha * scale_factor_from_dB)


def attenuation_coefficient_dBperm(freq, temp_deg, RH, pressure=np.array([1.01325e5], dtype=np.float64)):
    """ Implements the equations to calculate the attenuation coefficient and returns this for a given
    frequency, temperature and relative humidity.
    These formulae give estimates of the absorption of pure tones to an accuracy of
    ±10% for 0.05 < h < 5, 253 < temp (K) < 323, p0 < 200 kPa"""

    # Convert inputs into arrays so that we can call function on multiple inputs
    freq = np.array(freq, dtype=np.float64)
    temp_deg = np.array(temp_deg, dtype=np.float64)
    RH = np.array(RH, dtype=np.float64)
    pressure = np.array(pressure, dtype=np.float64)

    # Reference temperature 20 degrees celsius, 293.15 Kelvin
    T_0 = 293.15

    # Reference pressure is 1 atmosphere, or 1.01325e5 Pa
    P_0 = 1.01325e5

    # Relative pressure as this is easier for the equations
    p_r = pressure / P_0

    # Temperature in Kelvin
    T = temp_deg + 273.15

    # We only need h in this case, the percentage absolute humidity
    h = molar_concentration_water(temp_deg, RH)

    # These equations are derived from the physical principles behind attenuation i.e. molecular relaxation and kinetics etc
    f_r0 = p_r * 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    f_rN = p_r * (T / T_0) ** -0.5 * (9 + 280 * h * np.exp(-4.17 * ((T / T_0) ** (-1 / 3) - 1)))
    alpha = 8.686 * freq ** 2 * 1 / p_r * (1.84e-11 * (T / T_0) ** 0.5 + (T / T_0) ** (-5 / 2) * (
            0.01275 * np.exp(-2239.1 / T) / (f_r0 + freq ** 2 / f_r0) + 0.1068 * np.exp(-3352 / T) / (
            f_rN + freq ** 2 / f_rN)))

    return alpha


def attenuation_coeff(freq, temp_deg, rel_humid):
    """Code is meant to return the attenuation coefficient using the Stokes equation, it is completely wrong."""
    # Convert frequency in Hz to radians
    w = freq * 2 * np.pi

    # Convert temp to Kelvin
    temp_k = temp_deg + 273.15

    # Density
    abs_humid_frac, abs_humid_percent, density = humid(temp_deg, rel_humid)

    # Speed of sound
    speed_of_sound = sos(temp_deg, rel_humid)

    # Dynamic viscosity
    dyn_visc = 1.825e-5 * np.power(temp_k / 293.15, 0.7)
    # 0.001792 * np.power(np.e,(-1.94-4.8*273.16/temp_k+6.74*np.power(273.16/temp_k,2)))

    # 1.458e-6 * np.power(temp_k,1.5) / (temp_k + 110.4)

    coeff = 2 * dyn_visc * np.power(w, 2) / 3 / density / np.power(speed_of_sound, 3)

    return coeff  # Use the Stokes equation to calculate the attenuation coefficient


def vector2value(wind_vector, a, H):
    """
    Returns the wind speed in each direction from a to H. Returns this as a array with each sample, a, being
    row and columns being each microphone wind speed
    :param wind_vector: 1 x 3 array of wind vector
    :param a: n x 3 sound source location
    :param H: i x 3 microphone positions
    :return: n x i array of wind values
    """
    wind = np.zeros((a.shape[0], H.shape[0]))

    for i in range(H.shape[0]):
        line2sensor = H[i] - a
        line2sensor = line2sensor / np.linalg.norm(line2sensor, axis=1).reshape(-1, 1)
        wind[:, i] = np.dot(line2sensor, wind_vector)

    return wind


def dbAttenDivergence(x, A_0): return A_0 - 20 * np.log10(x)


""" Attenuation functions"""


def attenuation_absorption_at_x(freq, temp_deg, rel_humidity, x, A_0):
    """ Returns the dB of a signal at a distance x due to atmospheric absorption
    for a given frequency, temperature, relative humidity and initial dB level."""

    alpha = attenuation_coefficient_dBperm(freq, temp_deg, rel_humidity)

    return A_0 - alpha * x


def attenuation_divergence_at_r(A_0, r):
    """Calculates the attenuation due to divergence at a relative distance vector r from the source"""

    r = np.array(r, dtype=np.float64)

    return A_0 - 20 * np.log10(np.linalg.norm(r))


def attenuation_ground_at_r(A_0, r):
    """"Calculates the attenuation due to ground effects at a relative distance vector r from the source with strength A_0 dB"""

    # I think I'd like to do this using data rather than equations
    pass


def attenuation_turbulence_at_r(A_0, r):
    """Calculates the attenuation due to turbulence effects at a relative distance vector r from the source with source strength A_0 dB"""

    # Feel like this will be neglible, need to look into it
    pass


def attenuation_vegetation_at_r(A_0, r, frequency, type="Forest"):
    """"Calculates the attenuation due to vegetation effects at a relative distance vector r from the source with strength A_0 dB
    Equations obtained from """

    data_dict = {"Forest": 20} # Etc

    # There are some rough approximations
    if type == "Forest":
        return A_0 - 0.01 * frequency**(1/3) * r
    elif type == "Shrubs": # type must be shrubs
        return A_0 - (0.18*np.log10(frequency) - 0.31) * r
    elif type == None:
        return A_0
    else:
        print("Invalid vegetation type given")


def attenuation_wind_at_r(A_0, r):
    """"Calcualtes the attenuation due to wind effects at a relative distance vector r from the source with strength A_0 dB"""

    # Again probably data

    pass


def directivity_index_dB(r, directivity_cone_angle=45, percentage_power_in_cone=70.):
    """ Calculates the directivity index for a given source pressure and pressure at an angle from the directional source
    Essentially this returns the ratio of the pressure to the pressure of a non-directional source in dB
    Here the assumption is a uniform power within the cone angle, and a different uniform power outside
    We assume cone angle is a cone with vertex at the sound source and central axis the x-axis"""

    # Convert to numpy array if not already, remember r is relative to the source
    r = np.array(r, dtype=np.float64).reshape(-1, 3)
    # r coming in may be an array of values of shape num_samples x 3.

    # Cone angle into radians
    directivity_cone_angle = directivity_cone_angle / 180 * np.pi

    # Get the solid angle of the cone
    solid_angle = 2 * np.pi * (1 - np.cos(directivity_cone_angle))

    # We can get the directivity in the cone as
    directivity_cone = percentage_power_in_cone / 100 * 4 * np.pi / solid_angle

    # and outside the cone as
    directivity_outside_cone = (1 - percentage_power_in_cone / 100) * 4 * np.pi / (4 * np.pi - solid_angle)

    # Position r is within the cone angle if it is within the radius at that point
    # So we want to calculate the radius at a distance, r_x, from the tip of the cone
    radius = r[:, 0] * np.tan(directivity_cone_angle)

    # Now if the distance in the y and z direciton is less than the radius then it is within the cone
    radial_distance = np.linalg.norm(r[:, 1:], axis=1)

    print("Radial distance ", radial_distance.shape)
    print("R ", r.shape, r)

    ret = np.zeros_like(radial_distance)

    for i in range(r.shape[0]):
        if radial_distance[i] <= radius[i]:
            print("Within cone angle")
            ret[i] = 10 * np.log10(directivity_cone)
        else:
            print("Outside of cone angle")
            ret[i] = 10 * np.log10(directivity_outside_cone)
    return ret


class Signal():

    def __init__(self, signal_strength):
        self._signal_strength = signal_strength


def SPL_at_x(coeff, x, A_0, P_a=1.01325e5):
    # Pressure at reference/beginning
    P_0 = P_a * 10 ** (A_0 / 10)

    P_x = P_0 * 1 / x ** 2 * np.exp(-coeff * x)

    return P_x


def plot_attenuation_coefficient(tertiary_parameter, pressure=np.array([1.01325e5], dtype=np.float64),
                                 x_axis='Temperature', y_axis='RH', tertiary='Frequency'):
    temp_deg_values = 0
    freq_values = 0
    RH_values = 0

    axis_limits = {'Temperature': np.arange(-10, 40, 0.2), 'Frequency': np.arange(500, 10e3, 50),
                   'RH': np.arange(0, 100, 0.2)}
    values = {'Temperature': temp_deg_values, 'Frequency': freq_values, 'RH': RH_values}

    # Obtain X and Y in the meshgrid from our chosen axes
    values[x_axis], values[y_axis] = np.meshgrid(axis_limits[x_axis], axis_limits[y_axis])

    values[tertiary] = list(tertiary_parameter)

    # Always do 2 axes, can think of a way of improving this later
    fig, (ax1, ax2) = plt.subplots(1, 2)

    if tertiary == 'Frequency':
        Z = attenuation_coefficient_dBperm(values["Frequency"][0], values["Temperature"], values['RH'],
                                           pressure=pressure)
        contours1 = ax1.contourf(values[x_axis], values[y_axis], Z)
        fig.colorbar(contours1)
        Z = attenuation_coefficient_dBperm(values["Frequency"][1], values["Temperature"], values['RH'],
                                           pressure=pressure)
        contours2 = ax2.contourf(values[x_axis], values[y_axis], Z, levels=contours1.levels)
        fig.colorbar(contours2)


    elif tertiary == 'Temperature':
        Z = attenuation_coefficient_dBperm(values["Frequency"], values["Temperature"][0], values['RH'],
                                           pressure=pressure)
        contours1 = ax1.contourf(values[x_axis], values[y_axis], Z)
        Z = attenuation_coefficient_dBperm(values["Frequency"], values["Temperature"][1], values['RH'],
                                           pressure=pressure)
        ax2.contourf(values[x_axis], values[y_axis], Z, levels=contours1.levels)
        fig.colorbar(contours1)

    elif tertiary == 'RH':
        Z = attenuation_coefficient_dBperm(values["Frequency"], values["Temperature"], values["RH"][0],
                                           pressure=pressure)
        contours1 = ax1.contourf(values[x_axis], values[y_axis], Z)
        fig.colorbar(contours1)
        Z = attenuation_coefficient_dBperm(values["Frequency"], values["Temperature"], values["RH"][1],
                                           pressure=pressure)
        contours2 = ax2.contourf(values[x_axis], values[y_axis], Z, levels=contours1.levels)

    ax1.set_title("{} {}".format(tertiary, values[tertiary][0]))

    ax2.set_title("{} {}".format(tertiary, values[tertiary][1]))
   # ax2.colorbar(contours2)
    # ax1.set_xlabel("{}".format(x_axis))
    plt.xlabel(x_axis, position=(0, 0))
    ax1.set_ylabel(y_axis)
    plt.show()


def fix_parameter(parameter_to_fix='Temperature'):
    axis_limits = {'Temperature': np.arange(-10, 40, 0.5), 'Frequency': np.arange(500, 10e3, 100),
                   'RH': np.arange(0, 100, 0.5)}

    ymax_values = []
    ymin_values = []

    if parameter_to_fix == 'Frequency':
        X, Y = np.meshgrid(axis_limits['Temperature'], axis_limits['RH'])
    elif parameter_to_fix == 'Temperature':
        X, Y = np.meshgrid(axis_limits['Frequency'], axis_limits['RH'])
    elif parameter_to_fix == 'RH':
        X, Y = np.meshgrid(axis_limits['Temperature'], axis_limits['Frequency'])

    for value in axis_limits[parameter_to_fix]:

        if parameter_to_fix == 'Frequency':
            # Obtain the attenuation coefficient values
            attenuation_values = attenuation_coefficient_dBperm(value, X, Y)
        elif parameter_to_fix == 'Temperature':
            attenuation_values = attenuation_coefficient_dBperm(X, value, Y)
        elif parameter_to_fix == 'RH':
            attenuation_values = attenuation_coefficient_dBperm(Y, X, value)
        else:
            print("Incorrect paramter to fix")
            break

        max_attenuation = attenuation_values.max()
        min_attenuation = attenuation_values.min()

        ymax_values.append(max_attenuation)
        ymin_values.append(min_attenuation)

    plt.figure()
    plt.fill_between(axis_limits[parameter_to_fix], ymin_values, ymax_values)

    plt.title("Potential variation in atmospheric absorption for known {}".format(parameter_to_fix))
    plt.xlabel(parameter_to_fix)
    plt.ylabel("Atmospheric absorption (dB/m)")
    plt.show()

# fix_parameter("Temperature")
# fix_parameter("Frequency")
# fix_parameter("RH")
#
# plot_attenuation_coefficient([5e3, 8e3])
# plot_attenuation_coefficient([10, 30], x_axis="Frequency", y_axis="RH", tertiary="Temperature")
# plot_attenuation_coefficient([20, 80], x_axis="Temperature", y_axis="Frequency", tertiary="RH")



# x = np.linspace(1, 80, 100)
# y = dbAttenDivergence(x, A_0)
# plt.figure()
# plt.plot(x, y)
#
# plt.figure()
# y = atmospheric_absorption_at_x(freq, temp_deg, rel_humidity, x, A_0)
# plt.plot(x, y)
#
#
# #
# # #
# # # plt.figure()
# # # y = dB_at_x_3(attenuation_coeff(freq, temp_deg, rel_humidity), x, A_0)
# # # plt.plot(x, y)
# # #
# # # plt.figure()
# # # y = attenuation_eq(attenuation_coeff(freq, temp_deg, rel_humidity), A_0, x)
# # # plt.plot(x, y)
# # #
plt.show()

# print(attenuation_coefficient_dBperm(5000,20,80))
