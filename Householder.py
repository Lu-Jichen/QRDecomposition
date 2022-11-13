def householder_reduce(matrix_a):
    """
    Householder reduce for QR factorization.
    :param matrix_a:
    :return: matrix_q, matrix_r
    """

    # row num: n
    # col num: m
    n, m = matrix_a.shape

    matrix_r = matrix_a

    # H_1, H_2, ..., H_n
    householder_matrix_list = []

    # project each col onto standard basis
    for j in range(m):

        # Deal with un-reduced sub-matrix.
        sub_matrix = matrix_r[j:, j:]

        """ Get a_j """
        # a_j: column vector j
        a = np.reshape(sub_matrix[:, 0],
                       (len(sub_matrix), 1))

        # Check j-col
        if not np.nonzero(a)[0].any() or len(a) == 1:
            # All rested elements in col_j are zeros.
            continue

        """ Get v_j = a - |a| e """

        # 2-norm of vector a
        a_norm_2 = (np.sqrt(np.matmul(a.T, a)))

        # standard base e
        e = np.zeros_like(a)
        e[0] = 1

        # v = a - |a| e
        v = np.subtract(a, a_norm_2 * e)

        """ Get Householder matrix H_j"""

        # Household matrix H: I - 2 (vv')/(v'v)
        sub_matrix_h = np.identity(len(v)) - 2 * np.matmul(v, v.T) / np.matmul(v.T, v)

        # Augment Household matrix
        matrix_h = np.identity(n)
        matrix_h[j:, j:] = sub_matrix_h

        # Mapping current matrix
        matrix_r = np.matmul(matrix_h, matrix_r)

        # Store Household matrix
        householder_matrix_list.append(matrix_h)

    """ Reduce R matrix"""
    matrix_r = matrix_r[0:m]

    """ Compute Q' matrix """
    # Compute Q', where Q' = H_n ... H_2 * H_1 * I
    matrix_q = np.identity(n)

    for household_matrix in householder_matrix_list:
        matrix_q = np.matmul(household_matrix, matrix_q)

    """ Reduce Q matrix """
    matrix_q = np.transpose(matrix_q[0:m])

    return matrix_q, matrix_r
