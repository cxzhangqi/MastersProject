import numpy as np

def R(i, a, H):
    """
    Calculates the distance from source to microphone i.
    :param i: integer
    :param a: 3x1 np.darray, co-ordinates of source
    :param H: 4x3 np.darray, co-ordinates of microphones
    :return: ndarray, distance from microphone i to source a
    """
    z = H[i] - a.T

    return np.sqrt(np.dot(z,z.T))

def m(i, j, a, H):
    """

    :param i: integer, denotes microphone
    :param j: integer, denotes co-ordinate x=0, y=1 or z=2
    :param a: 3x1 np.darray, co-ordinates of source
    :param H: 4x3 np.darray, co-ordinates of microphones
    :return: Returns element i,j of matrix M
    """
    if R(i, a, H) == 0:
        return 0
    else:
        return np.divide(np.subtract(H[i][j], a[0][j]), R(i, a, H))

def d(i, j, a, H, v_sound):
    """

    :param i: integer, denotes microphone
    :param j: integer, denotes co-ordinate x=0, y=1 or z=2
    :param a: 3x1 np.darray, co-ordinates of source
    :param H: 4x3 np.darray, co-ordinates of microphones
    :param v_sound: float, speed of sound
    :return: Returns element i,j or matrix D
    """
    if R(i, a, H) == 0:
        return 0
    else:
        return np.divide(np.subtract(H[i][j], a[0][j]), np.multiply(v_sound, R(i, a_x, a_y, a_z, H)))

def dDt_dv_ij(i, j, a, H, v_sound):
    """

    :param i:
    :param j:
    :param a:
    :param H:
    :param v_sound:
    :return:
    """
    h_i = np.sqrt(np.power((a_x - H[i][0]), 2) + np.power((a_y - H[i][1]), 2))

    h_j = np.sqrt(np.power((a_x - H[j][0]), 2) + np.power((a_y - H[j][1]), 2))

    return np.multiply(np.divide(a_z, v_sound), np.subtract(np.divide(1, np.sqrt(np.power(a_z, 2) + np.power(h_i, 2))),
                                                            np.divide(1, np.sqrt(np.power(a_z, 2) + np.power(h_j, 2)))))

