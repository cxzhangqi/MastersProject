import numpy as np
import matplotlib.pyplot as plt

def humid(temp_deg, rel_humid):
    vapor_pressure = 1000 * 0.61121 * np.exp((18.678 - temp_deg / 234.5) * (
            temp_deg / (257.14 + temp_deg)))  # Approximate vapor_pressure with the Buck equation

    p_v = rel_humid / 100 * vapor_pressure

    p_d = 1.013e5 - p_v

    density_humid = (p_d * 0.028966 + p_v * 0.018016) / 8.314 / (temp_deg + 273.15)

    abs_humidity = 0.01 * rel_humid * vapor_pressure / 1.013e5  # Absolute humidity value based on temperature (for vapor pressure) and relative humidity

    return abs_humidity, density_humid


def sos(temp_deg, rel_humid):
    temp_k = temp_deg + 273.15

    abs_humidity, density = humid(temp_deg, rel_humid)

    spec_heat_ratio = (7 + abs_humidity) / (
            5 + abs_humidity)  # Specific heat ratio is dependent on the abs humidity, it is 1.4 at 0 humidity

    molar_mass = 28.966 * (
            1 - abs_humidity) + abs_humidity * 18.01528  # Get the new molar mass with that fraction of water

    sos = np.sqrt(spec_heat_ratio * 8310 * temp_k / molar_mass)

    return sos


def attenuation_coeff(freq, temp_deg, rel_humid):
    # Convert frequency in Hz to radians
    w = freq * 2 * np.pi

    # Convert temp to Kelvin
    temp_k = temp_deg + 273.15

    # Density
    abs_humid, density = humid(temp_deg, rel_humid)

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

def attenuation_eq(coeff, A_0, x):
    # A = A_0/np.power(x,2)*np.power(np.e,-coeff*x/2)
    #
    # A = 10*np.log10(A_0/np.power(x,2)) - coeff*np.power(x,2)/2*10*np.log10(np.e)
    #
    # A = A_0 + A

    y = - 20 * np.log10(x) + A_0*np.exp(-coeff * x)  # NEED TO CHECK IF THIS IS CORRECT

    # x_solve = sympy.symbols('x_solve')
    #
    # expr = A_0 - 20*sympy.log(x_solve,10) + A_0*(1 - np.power(np.e,-coeff*x_solve/2))
    #
    # zero_crossing = solve(A_0 - 20*np.log10(x_solve) + A_0*(1 - np.power(np.e,-coeff*x_solve/2)),x)

    return y  # , zero_crossing[0]

def dB_at_x(coeff, x, A_0):

    return 10*np.log10((1 / x**2)) - np.exp(-coeff * x)

def dB_at_x_2(coeff, x, A_0):

    return A_0 + 20*np.log10(1 / x) - 10 * coeff * x * np.log10(np.e)

def dB_at_x_3(coeff, x, A_0):

    return A_0 - 10 * coeff * x * np.log10(np.e) - 20*np.log10(x)

A_0 = 85
freq = 5e3  #Hz
temp_deg = 20
rel_humidity = 85 #%
x = np.linspace(1, 80, 100)
y = dB_at_x(attenuation_coeff(freq, temp_deg, rel_humidity), x, A_0)
plt.figure()
plt.plot(x, y)

plt.figure()
y = dB_at_x_3(attenuation_coeff(freq, temp_deg, rel_humidity), x, A_0)
plt.plot(x, y)

plt.figure()
y = attenuation_eq(attenuation_coeff(freq, temp_deg, rel_humidity), A_0, x)
plt.plot(x, y)

plt.show()
