import numpy as np
import matplotlib.pyplot as plt

def R(a, H, model='3D'):
    """
    Calculates the distance from source to microphone i.
    :param i: integer
    :param a: 3x1 np.darray, co-ordinates of source
    :param H: 4x3 np.darray, co-ordinates of microphones
    :return: 4x1 ndarray, distance from microphones to source a
    """
    if model == '3D':
        z = H - a

        return np.sqrt(np.sum(np.square(z), axis=0))
    else:
        z = H[:][:-1] - a[:-1]
        return np.sqrt(np.sum(np.square(z), axis=0))  # Could also dot Z with Z.T, don't know which is better



def M(a, H):
    m = np.divide((H - a), R(a, H), out=np.zeros_like((H - a), where=R(a, H)!=0))    # When R returns 0 this will just put 0 in array, rather than a divide by 0 error
    return np.concatenate(m, np.ones((4,1)), axis=1)

def D(a, H, v_sound):
    d = np.divide((H - a), v_sound * R(a, H), out=np.zeros_like((H - a), where=R(a,H) != 0))  # When R returns 0 this will just put 0 in array, rather than a divide by 0 error
    return np.concatenate(d, np.ones((4, 1)), axis=1)

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

def E_sos(a, H, dc, v_sound):
    """
    Returns a 4x1 matrix containing the position estimate errors for error in speed of sound, dc
    First index is error in x
    Second index is error in y
    Third index is error in z
    :param a: 3x1 ndarray, co-ordinates of source
    :param H: 4x3 ndarray, co-ordinates of microphones
    :param dc: float, error in speed of sound estimate
    :return: 4x1 ndarray containing position estimate errors DAx, DAy, DAz and ...
    """

    M_matrix = M(a, H)

    M_inv = np.linalg.inv(M_matrix, out=np.zeros_like(M_matrix), where=np.linalg.cond(M_matrix) < 1/sys.float_info.epsilon)

    T = R(a, H) / v_sound

    return np.multiply(np.matmul(M_inv, T), dc)  # This is a 4x1 array with all the errors in position and that final Rm

def E_time(a_x, a_y, a_z, H, DT, v_sound):
    """
        Returns a 4x1 matrix containing the position estimate errors for error in time of arrivals to each microphone, 4x1 matrix ToA
        First index is error in x
        Second index is error in y
        Third index is error in z
        :param a: 3x1 ndarray, co-ordinates of source
        :param H: 4x3 ndarray, co-ordinates of microphones
        :param DT: 4x1 ndarray, error in speed of sound estimate
        :return: 4x1 ndarray containing position estimate errors DAx, DAy, DAz and ...
        """
    D = np.zeros((4, 4))

    for i in range(4):
        D[i][3] = 1
        for j in range(3):
            D[i][j] = d(i, j, a_x, a_y, a_z, H, v_sound)

    if np.linalg.det(D) == 0:  # Need to figure out a better way of handling this. For now just do this?
        print("\n We got here!")

    D_inv = np.linalg.inv(D)

    return np.matmul(D_inv, DT)  # Should be a 3 x 1 array


# plot hyperbola(a,)
# For now, just do it between two microphones rather than getting into a matrix.
def E_2D(a_x, a_y, a_z, H, v_sound, i, j):
    dDt_dv = dDt_dv_ij(i, j, a_x, a_y, a_z, H, v_sound)

    return np.multiply(a_z, dDt_dv)  # Returns the change in arrival time between microphones i and j

def calculate_error_indicator(error_matrix,xrange,yrange):
    # Integrate
    sum = 0

    for i in range(5, 15):
        for j in range(5, 15):
            sum += error_matrix[j][i]

    # Return error square divided by area of the box
    return np.divide(np.power(sum, 2), 100)
