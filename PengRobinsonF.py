import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.linalg as lin
from numba import jit

#1) рассмотрим бинарный газ полностью расстворенный в жидкой фазе воды
# для расчетов исспользуем уравнение Пенга - Робинсона


#begin values and constant

N = 2

p = 300.0 * 10 ** 5

T = 331.0

R = 8.31

y0 = [ 0.3, 0.3 ]

V0 = 0.3

def delta(i, j):
    if i == j:

        return 0.0

    elif i != j:

        return 1.0


table = pd.DataFrame(
    {'z(%)': [100.0 - 48.999, 48.999, 1.089, 78.64, 8.0958, 3.8572, 2.195, 0.87999, 2.854, 1.564, 0.13522],
     'M(g/mole)': [18.01, 28.013, 44.01, 16.043, 30.07, 44.097, 58.124, 72.151, 106.04, 212.27, 403.47],
     'Pcr(bar)': [221.19, 33.944, 73.866, 46.042, 48.839, 42.455, 37.47, 33.589, 23.999, 17.637, 10.039],
     'Tcr(K)': [647.0, 126.2, 304.7, 190.6, 305.43, 369.8, 419.5, 465.9, 550.78, 829.01, 829.73],
     '$\Omega_A$': [0.45724, 0.45724, 0.45724, 0.45724, 0.45724, 0.45724, 0.45724, 0.45724, 0.37058, 0.33425, 0.51705],
     '$\Omega_B$': [0.077796, 0.077796, 0.077796, 0.077796, 0.077796, 0.077796, 0.077796, 0.077796, 0.061461, 0.063713,
                    0.075328],
     '$\omega$': [0.01300, 0.04, 0.225, 0.013, 0.0986, 0.1524, 0.1956, 0.2413, 0.25344, 0.43207, 0.93158],
     'c-shift': [-0.00386, -0.00316, -0.00114, -0.00386, -0.00314, -0.018358, -0.00555, -0.00517, -0.00767, -0.00539,
                 -0.00416]})

table.index = ['H20', 'N2', 'CO2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6+', 'C11+', 'C27+']

jk = np.zeros((N, N))

tr = table['Tcr(K)'][:2] / T

pr = table['Pcr(bar)'][:2] / p

omega_a = table['$\Omega_A$'][:2] * (
            1 + (0.37464 + 1.54226 * table['$\omega$'][:2] - 0.2669 * table['$\omega$'][:2] ** 2) * (
                1 - np.sqrt(tr))) ** 2

A = omega_a * (pr / tr ** 2)

B = table['$\Omega_B$'][:2] * (pr / tr)

A2 = np.zeros((N, N))

jk = np.zeros((N, N))

for i in range(N):

    for j in range(N):
        jk[i][j] = delta(i, j) * (-0.06 + 2.06 * T + 0.659 * T ** 2)

        A2[i][j] = (1.0 - jk[i][j]) * np.sqrt(A[j] * A[i])

#################################################################################
#################################################################################


@jit
def log_f(yi, x, p, Z, Am, ai, Bm, Bi):
    Cm = (1.0 + 2.0 ** 0.5) * Bm
    Dm = (1.0 - 2.0 ** 0.5) * Bm

    ci = (1.0 + 2.0 * 0.5) * Bi
    di = (1.0 - 2.0 ** 0.5) * Bi

    return np.log(p * yi) - np.log(Z - Bm) - Am / (Cm - Dm) * (2.0 * sum(x * ai) / Am - Bi / B) * np.log(
        (Z + Cm) / (Z + Dm)) + Bi / (Z - Bm) - Am / (Cm - Dm) * (di / (Z + Cm) - di / (Z + Dm))

@jit
def Si(A, x):
    global N

    S_return = 0.0

    for i in range(N):
        S_return = S_return + A[i] * x[i]

    return S_return


@jit
def A_res(A, x):
    global N

    A_return = 0.0

    for i in range(N):

        for j in range(N):
            A_return += A[i][j] * x[i] * x[j]

    return A_return


@jit
def B_res(x):
    global N

    global B

    B_return = 0.0

    for i in range(N):
        B_return = B_return + B[i] * x[i]

    return B_return


@jit
def log_f_(x, p, Z, Am, Bm, a, B):

    global N, T, R

    def delta(i, j):

        if i == j:

            return 1.0

        elif i != j:

            return 0.0

    b = B

    d = (1 - 2.0 ** 0.5) * b

    c = (1 + 2.0 ** 0.5) * b

    Cm = (1 + 2.0 ** 0.5) * Bm

    Dm = (1 - 2.0 ** 0.5) * Bm

    cm = Cm

    dm = Dm

    M = np.zeros((N, N))

    dFdz = 3.0 * Z ** 2 + 2.0 * Z ** 2 * (Cm + Dm - Bm - 1) + (Am + Bm * Cm + Cm * Dm - Bm * Dm - Cm - Dm)

    dFdx = np.zeros((N, N))

    Q = np.zeros((N, N))

    dlnfdx = np.zeros((N, N))

    for i in np.arange(0, N, 1):

        for j in np.arange(0, N, 1):
            dFdx[i][j] = p / (R * T) * (2 * (1 - 2.0 ** 0.5) * b[j] * Z ** 2 + (
                        2.0 / R / T * sum(x * a[i]) - Cm * b[j] - Bm * c[j] + Dm * c[j] + Cm * d[j] - Dm * d[j] - d[j] -
                        c[j]) * Z
                                        - Cm * Dm * b[j] - Bm * Dm * c[j] - Bm * Cm * d[j] - Dm * c[j] - Cm * d[
                                            j] - Am * b[j] - 2.0 * Bm / R / T * sum(x * a[i]))

            dzdx = dFdx / dFdz

            M[i][j] = 1.0 / (Cm - Dm) * (2.0 * p / R ** 2 / T ** 2.0 * sum(x * a[i]) - Am / (Cm - Dm) * (c[j] - d[j]))

            Q[i][j] = (2.0 * a[i][j] / Am - 4.0 * sum(x * a[i]) * sum(x * a[i]) / Am ** 2 + (c[i] - d[i]) * (
                        c[j] - d[j]) / (Cm - Dm) ** 2) * np.log((Z + Cm) / (Z + Dm)) + (
                                  2.0 * sum(x * a[i]) / Am - (c[j] - d[j]) / (Cm - Dm)) * (
                                  1.0 / (Z + Cm) * (dzdx + c[j]) - 1.0 / (Z + Dm) * (dzdx + d[j])) - c[j] / (
                                  Z + Cm) ** 2 * (dzdx + c[j]) + d[j] / (Z + Dm) ** 2 * (dzdx + d[j])

            dlnfdx[i][j] = delta(i, j) / x[j] - 1.0 / (Z - Bm) * (dzdx - b[j]) * (1.0 + B[i] / (Z - Bm))
            - M[i][j] * ((2.0 * sum(x * a[i]) / Am - (c[i] - d[i]) / (cm - dm)) * np.log((Z + Cm) / (Z + Dm))
                         + (c[i] / (Z + Cm) - d[i] / (Z + Dm))) - Am / (Cm - Dm) * Q[i][j]

    return dlnfdx


@jit
def J( y, p, z, Am, Bm, a, B, V):

    global log_f_

    global N

    Result = np.zeros((N, N))

    dlnfdy = log_f_(y, p, z, Am, Bm, a, B)

    dlnfdx = log_f_((z - y * V) / (1.0 - V), p, z, Am, Bm, a, B)

    for i in np.arange(0, N, 1):

        Result[i][0] = - 1.0 / ( 1.0 - V ) * sum(dlnfdx[i] * (y - (z - y * V) / (1.0 - V)))

        for j in np.arange(1, N, 1):
            Result[i][j] = V / (1.0 - V) * dlnfdx[i][0] - V / (1.0 - V) * dlnfdx[i][j] + dlnfdy[i][0] - dlnfdy[i][j]

    return Result

@jit
def Newton(y0, V0):
    global N

    global log_f, J

    global A_res

    global B_res

    global A2, B, table, p

    eps = 1.0e-8

    S = np.zeros(N + 1)

    F = np.zeros(N + 1)

    uslovie = 0.0

    S[0] = V0

    for i in np.arange(1, N + 1, 1):
        S[i] = y0[i - 1]

    print('Yes_S')

    for i in np.arange(1, N + 1, 1):
        print("Yes", A_res(A2, S[1:]), B_res(S[1:]))

        uslovie += (log_f(S[i], S[1:], p, table['z(%)'][i] / 100.0, A_res( A2, S[1:] ), A2[i - 1], B_res(S[1:]), B[i - 1] ) / log_f( (table['z(%)'][i] / 100.0 - S[i] * S[0]) / (1.0 - S[0]),  (table['z(%)'][i] / 100.0 - S[1:] * S[0]) / (1.0 - S[0]) , p, table['z(%)'][i] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]) - 1.0 ) ** 2

        print('Yes!')

    print("Yes")

    F[0] = - (log_f((table['z(%)'][0] / 100.0 - S[1] * S[0]) / (1.0 - S[0]), (table['z(%)'][0] / 100.0 - S[1:] * S[0]) / (1.0 - S[0]),  p, table['z(%)'][0] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]) - log_f(S[1], S[1:], p, table['z(%)'][0] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]))

    for i in np.arange(1, N + 1, 1):
        F[i] = - (log_f((table['z(%)'][i - 1] / 100.0 - S[i] * S[0]) / (1.0 - S[0]), (table['z(%)'][i - 1] / 100.0 - S[1:] * S[0]) / (1.0 - S[0]),  p, table['z(%)'][i - 1] / 100.0,
                        A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]) - log_f(S[i], S[1:], p,
                                                                                     table['z(%)'][i - 1] / 100.0,
                                                                                     A_res(A2, S[1:]), A2[i - 1],
                                                                                     B_res(S[1:]), B[i - 1]))

    while np.sqrt(uslovie) >= eps:

        S = lin.solve(J(p, table['z(%)'][i] / 100.0, A_res(A2, S[1:]), B_res(S[1:]), A2, B, S[0]), F)

        uslovie = 0.0

        for i in np.arange(1, N + 1, 1):
            uslovie += (log_f(S[i], p, table['z(%)'][i] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]),
                              B[i - 1]) / log_f((table['z(%)'][i] / 100.0 - S[i] * S[0]) / (1.0 - S[0]), p,
                                                table['z(%)'][i] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]),
                                                B[i - 1]) - 1) ** 2

        F[0] = - (log_f((table['z(%)'][0] / 100.0 - y0[0] * S[0]) / (1.0 - S[0]), (table['z(%)'][0] / 100.0 - y0 * S[0]) / (1.0 - S[0]), p, table['z(%)'][0] / 100.0,
                        A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]) - log_f(y0[0], y0, p, table['z(%)'][1] / 100.0,
                                                                                     A_res(A2, S[1:]), A2[i - 1],
                                                                                     B_res(S[1:]), B[i - 1]))

        for i in np.arange(1, N + 1, 1):
            F[i] = - (log_f( (table['z(%)'][i - 1] / 100.0 - S[i] * S[0]) / (1.0 - S[0]), (table['z(%)'][i - 1] / 100.0 - S[1:] * S[0]) / (1.0 - S[0]), p,
                            table['z(%)'][i - 1] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]) - log_f(
                S[i], S[1:], p, table['z(%)'][i - 1] / 100.0, A_res(A2, S[1:]), A2[i - 1], B_res(S[1:]), B[i - 1]))

    return S









