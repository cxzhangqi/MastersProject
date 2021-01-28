import numpy as np

def E_sos(a, H, dc):
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
