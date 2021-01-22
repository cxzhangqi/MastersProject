import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

# Definition of microphone co-ordinates in 2D
h1 = [0, 0]
h2 = [10, 0]
h3 = [5, 10]
h4 = [10,10]

# Put microphone co-ordinates in a matrix, H
H = np.array([h1, h2, h3])

# Set variables
v_sound = 343.382902  # Speed of sound
a_xrange = 30  # Maximum range in x for position of sound source, a
a_xmin = 5 - a_xrange / 2
a_yrange = 30  # Maximum range in y for position of sound source, a
a_ymin = np.sqrt(75) / 2 - a_yrange / 2
dc = 5  # Speed of sound error


# increment = 1
#
# no_steps_x = int(max_a_x*2/increment + 1)
# no_steps_y = int(max_a_y*2/increment + 1)
#
# x_range = np.arange((5 - max_a_x), (5+max_a_x) , increment)
# y_range = np.arange((math.sqrt(75)/2 - max_a_y), (math.sqrt(75)/2 + max_a_y), increment)


def R(i, a_x, a_y, H):
    """
    Definition of R the distance from the sound source, a, to microphone i in 2D
    """
    return np.sqrt(np.power((H[i][0] - a_x), 2) + np.power((H[i][1] - a_y), 2))


def m(i, j, a_x, a_y, H):
    """
    Returns the value of M[i,j] with sound source, a, and microphone positions H.

    """
    if j == 0:
        choice = a_x
    else:
        choice = a_y

    if R(i, a_x, a_y, H) == 0:
        return 0
    else:
        return np.divide(np.subtract(H[i][j], choice), R(i, a_x, a_y, H))


def E(a_x, a_y, H, dc):
    """
    Returns a 3x1 matrix containing the position estimate errors for error in speed of sound, dc
    First index is error in x
    Second index is error iny
    Third index is...
    :param a:
    :param H:
    :return:
    """
    # m11 = m(0, 0, a_x, a_y, H)
    # m12 = m(0, 1, a_x, a_y, H)
    #
    # m21 = m(1, 0, a_x, a_y, H)
    # m22 = m(1, 1, a_x, a_y, H)
    #
    # m31 = m(2, 0, a_x, a_y, H)
    # m32 = m(2, 1, a_x, a_y, H)

    assert R(0,0,0,H) == 0
    assert R(1,0,0,H) == 10

    assert m(1,0,0,0,H) == 1

    M = np.zeros((3, 3))

    for i in range(3):
        M[i][2] = 1
        for j in range(2):
            M[i][j] = m(i, j, a_x, a_y, H)

    M_inv = np.linalg.inv(M)

    T = np.array([[np.divide(R(0, a_x, a_y, H), v_sound)], [np.divide(R(1, a_x, a_y, H), v_sound)],
                  [np.divide(R(2, a_x, a_y, H), v_sound)]])

    return np.multiply(np.matmul(M_inv, T), dc)  # This is a 4x1 array


absolute_error_sos = np.zeros((a_xrange, a_yrange))

x = np.arange(a_xmin, a_xrange + a_xmin, 1)
y = np.arange(a_ymin, a_yrange + a_ymin, 1)

for i in range(a_xrange):
    for j in range(a_yrange):
        a_x = a_xmin + i
        a_y = a_ymin + j
        a = [a_x, a_y]
        print(a)

        error_matrix = E(a_x, a_y, H, dc)

        absolute_error_sos[j][i] = np.sqrt(math.pow(error_matrix[0][0], 2) + math.pow(error_matrix[1][0], 2))

# Define variables delta t
Dt1 = [3]
Dt2 = [0]
Dt3 = [3]
DT = np.array([Dt1, Dt2, Dt3])


def d(i, j, a_x, a_y, H):
    if j == 0:
        choice = a_x
    else:
        choice = a_y

    if R(i, a_x, a_y, H) == 0:
        return 0
    else:
        return np.divide((H[i][j] - choice), np.multiply(v_sound, R(i, a_x, a_y, H)))


def E_time(a_x, a_y, DT, H):

    D = np.zeros((3, 3))

    for i in range(3):
        D[i][2] = 1
        for j in range(2):
            D[i][j] = d(i, j, a_x, a_y, H)

    D_inv = np.linalg.inv(D)

    return np.matmul(D_inv,DT)  # Should be a 3 x 1 array

absolute_error_time = np.zeros((a_xrange, a_yrange))

for i in range(a_xrange):
    for j in range(a_yrange):
        a_x = a_xmin + i
        a_y = a_ymin + j
        a = [a_x, a_y]

        error_matrix = E_time(a_x, a_y, DT, H)

        absolute_error_time[j][i] = np.sqrt(math.pow(error_matrix[0][0], 2) + math.pow(error_matrix[1][0], 2))

# print(absolute_error)

absolute_error = absolute_error_sos + absolute_error_time

X, Y = np.meshgrid(x, y, indexing='xy')

plt.figure(1)

# Z = np.sqrt(np.power(E_vec(X, Y, H, dc)[0][0], 2) + np.power(E_vec(X, Y, H, dc)[1][0], 2))

contours = plt.contourf(X, Y, absolute_error_sos,levels=[0.5,1])
#plt.clabel(contours, inline=True, fontsize=10)
plt.colorbar()

# Draw the sensor positions in
plt.gca().add_patch(plt.Polygon(H, facecolor=None,fill=False))

plt.figure(2)

plt.contourf(X,Y,absolute_error_time, 400)
plt.gca().add_patch(plt.Polygon(H, facecolor=None,fill=False))
plt.colorbar()

plt.show()