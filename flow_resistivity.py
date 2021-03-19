import numpy as np
from scipy.special import erfc
from enviro_calc import sos, sos_old, humid
import matplotlib.pyplot as plt

""" This file implements the equations found in Embleton 1983, Effective flow resisitivity of ground surfaces
determined by acoustical measurements. These equations calculate the expected transmission spectra for given
geometry of source, receiver and separation for different frequencies etc. These values, whilst not fully
descriptive of reality, do explain a lot of the perceived spectra and as such will serve as a useful tool
for predicting spectra. We note that the acoustic impedance of air CHANGES with temperature and humidity
and thus this will also affect these graphs."""

temp = 20
RH = 80

# Properties of air
c_air = sos_old(temp, RH)
_, _, density_air = humid(temp, RH)

# Characteristic acoustic impedance
Z_0 = density_air * c_air # kg/m/m/s

def calculate_reflection(source_position, receiver_position):
    """Calculates reflected distance from receiver to ground and reflection angle"""

    # The horizontal distance between them
    b = np.linalg.norm(source_position[:-1] - receiver_position[:-1])

    # Ratio of z1 and z2 for similar triangles
    ratio = source_position[-1] / receiver_position[-1]

    # Second right angle triangle hypotenuse
    hypot_2 = np.sqrt(b ** 2 / (1 + ratio) ** 2 + receiver_position[-1] ** 2)

    # First hypotenuse from similar triangles
    hypot_1 = hypot_2 * ratio

    # Total distance is the sum of the two
    r2 = hypot_1 + hypot_2

    # Now we also will want the reflection angle
    reflection_angle = np.arcsin(receiver_position[-1] / hypot_2)

    return r2, reflection_angle

def numerical_distance(frequency, eff_flow_resistivity, source_position, receiver_position, k1, Z1, R_p):
    """ Returns how the distance as a multiple of wavelengths"""

    r2, reflection_angle = calculate_reflection(source_position, receiver_position)

    Z2 = acoustic_impedance(frequency, eff_flow_resistivity, Z1)

    k2 = wave_number(frequency, eff_flow_resistivity, k1)

    first_term = 2*k1*r2/(1-R_p)**2

    second_term = (Z1 / Z2)**2

    third_term = 1 - k1**2 / k2**2 * np.cos(reflection_angle)**2

    return 1j * first_term * second_term * third_term

def plane_wave_reflection_coefficient(frequency, eff_flow_resistivity, k1, Z1, reflection_angle):

    Z2 = acoustic_impedance(frequency, eff_flow_resistivity, Z1)

    k2 = wave_number(frequency, eff_flow_resistivity, k1)

    numerator = Z2 * np.sin(reflection_angle) - Z1 * (1 - k1**2 / k2**2 * np.cos(reflection_angle)**2)**0.5

    denominator = Z2 * np.sin(reflection_angle) + Z1 * (1 - k1**2 / k2**2 * np.cos(reflection_angle)**2)**0.5

    return numerator / denominator

def ground_wave_function(frequency, eff_flow_resistivity, source_position, receiver_position, k1, Z1, R_p):

    w = numerical_distance(frequency, eff_flow_resistivity, source_position, receiver_position, k1, Z1, R_p)

    return 1 + 1j * np.pi**0.5 * w**0.5 * np.e**(-w) * erfc(-1j * np.sqrt(w))

def acoustic_impedance(frequency, eff_flow_resistivity, Z1):

    R2 = Z1 * (1 + 9.08 * (frequency / eff_flow_resistivity)**-0.75)

    X2 = Z1 * (11.9 * (frequency / eff_flow_resistivity)**-0.73)

    return R2 + X2 * 1j

def wave_number(frequency, eff_flow_resistivity, k1):

    alpha = k1 * (1 + 10.8*(frequency / eff_flow_resistivity)**-0.7)

    beta = k1 * (10.3 * (frequency / eff_flow_resistivity)**-0.59)

    return alpha + beta*1j

def pressure_at_receiver(frequency, eff_flow_resistivity, source_position, receiver_position, temp = 0, RH = 0):

    _, _, density_air = humid(temp, RH)
    c_air = sos_old(temp, RH)
    print(density_air, c_air)

    # Get impedances for air (1) and ground (2)
    Z1 = c_air * density_air
    print("Z1: ", Z1)

    # Get distances r1 (direct) and r2 (reflected)
    r1 = np.linalg.norm(source_position - receiver_position)
    r2, reflection_angle = calculate_reflection(source_position, receiver_position)

    print("r1: ", r1)
    print("r2: ", r2)

    # Obtain wave numbers for sound wave in air and in the ground
    k1 = 2*np.pi*frequency / c_air
    print("k1: ", k1)

    # Obtain plane-wave reflection coefficient
    R_p = plane_wave_reflection_coefficient(frequency, eff_flow_resistivity, k1, Z1, reflection_angle)

    print("R_p: ", R_p)

    # Obtain the ground wave function evaluated at the numerical distance
    F = ground_wave_function(frequency, eff_flow_resistivity, source_position, receiver_position, k1, Z1, R_p)

    print("F: ", F)

    first_term = np.e**(1j*k1*r1) / (k1 * r1)

    second_term = R_p * (np.e**(1j*k1*r2) / (k1 * r2))

    third_term = (1 - R_p) * F * np.e**(1j * k1 * r2) / (k1 * r2)

    return first_term + second_term + third_term

# Point source of sound in 3D
a = np.array([0, 0, 0.31], dtype=np.float64)

# Arbitrary separation in x and y
x_separation = 15.2
y_separation = 0

# Receiver position
m = np.array([x_separation, y_separation, 1.22], dtype=np.float64)

# Distance from source to receiver
r1 = np.linalg.norm(a - m)

# Distance from source to receiver via ground
r2, reflection_angle = calculate_reflection(a, m)

sigma = 10
frequency = np.arange(100, 10e3, 1)
#y = pressure_at_receiver(frequency, sigma, a, m)
logy = 10*np.log10(pressure_at_receiver(frequency, sigma, a, m))

plt.figure(1)
plt.plot(frequency, logy.real)
plt.xscale("log")
plt.show()

# x = np.linspace(-4,60,100)
# y = erfc(x)
#
# plt.figure(1)
# plt.plot(x, y)
# plt.show()