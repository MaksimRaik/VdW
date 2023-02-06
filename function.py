import numpy as np
from numba import jit
#файл содержит функции основных уравнений состояния
# и термодинамических величин для газа Ван дер Ваальса


R = 25 / 3.0

a = 27 / 64.0 * ( R**2 * 647.13**2 / 22.055 / 10.0**6 )

b = 1 / 8.0  * ( R * 647.13 / 22.055 / 10.0**6 )

k = 1.38 * 10 ** -23

eps = 10 ** -7

Number = 4000

T = np.asarray( [ 545 + 10 * i for i in range(5) ] )

V = np.asarray([b + i * eps for i in range(1, Number)])

color = [ 'r-', 'b-', 'g-', 'c-', 'k-', 'm-' ] #цвета для построения графиков, 6 штук

x1 = 0.005

x2 = 500

v_i = 0

@jit
def Van_der_Vaals(T, V):

    global a, b, R

    return R * T  / ( V - b ) - a / V ** 2

@jit
def Si( V ):

    global a, b

    return ( V - 2 * b ) * a / V ** 3

@jit
def free_energy( T, V ):

    global a, b, R

    return - R * T * ( 1.0 + np.log( ( V - b ) * T ** ( 3 / 2.0 ) ) ) - a / V

@jit
def chemical_potencial( V, T ):

    global a, b, R

    nu = -R * T * ( 1.0 + np.log( ( V - b ) * T ** ( 3.0 / 2.0 ) ) ) + R * T * V / ( V - b ) - 2 * a / V

    return nu

