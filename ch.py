import function as f
import numpy as np
from numba import jit
# 1) Обозначения:
# f(x) - функция
# f(x)_ - производная функции f(x)
# f(x)__ - вторая производная функции f(x)

max_min = 0.0

@jit
def otrezok( f, a, b, mass ): #нахождение минимума методлом деления отрезка пополам

    x00 = []

    for i in mass:

        xa = a

        xb = b

        x0 = ( xa + xb ) / 2.0

        #print( x0 )

        while np.fabs( f( x0, i ) ) >= 8 * 10 ** -5:

            if f( xa, i ) * f( x0, i ) <= 0.0:

                xb = x0

                x0 = ( xa + xb ) / 2.0

                #print( x0 )

            elif f( xb, i ) * f( x0, i ) < 0.0:

                xa = x0

                x0 = ( xa + xb) / 2.0

                #print( x0 )

        #print( x0 )

        x00.append( x0 )

    return np.asarray( x00 )

@jit
def VdW_Si( V, T ): # вспомогательная функция для нахождения граничных точек
    # неустойчивой зоны

    return f.Van_der_Vaals( T, V ) - f.Si( V )

@jit
def VdW_Si_( V, T ):

    return - f.R * T / ( V - f.b ) ** 2 + f.a / V ** 3 + 3 * ( V - 2 * f.b ) * f.a / V ** 4

@jit
def critical_point( V1, V2, T ): #функция нахождения граничных точек неустойчивой зоны

    global otrezok, VdW_Si

    return otrezok( VdW_Si, V1, V2, T )


def chemical_point_equlibrium( V, T ): # T - число, V - число, возвращает массив значений с вычитом максимального или минимального элемента.

    global max_min

    return f.chemical_potencial( V, T ) - max_min # результат используется в основном для функции otrezok( f, x1, x2, mass )

def equlibrium(): # нахождение границ областей с одинаковыми химическими потенциалами

    global max_min, critical_point, otrezok, chemical_point_equlibrium

    v_equlibrium = []

    v2_equlibrium = []

    V1_mass = critical_point( f.V[0], f.V[400], f.T )

    V2_mass = critical_point( f.V[440], f.V[ f.Number - 3 ], f.T )

    for j in f.T:

        Vch = np.linspace( f.V[1], V1_mass[ list(f.T).index( j ) ], f.x2 )

        Vch2 = np.linspace( V2_mass[ list(f.T).index( j ) ], f.x1, f.x2 )

        max_min = max( f.chemical_potencial( Vch2, j ) )

        v_equlibrium.append( otrezok( chemical_point_equlibrium, Vch[0], Vch[-1], f.T)[ list(f.T).index( j ) ] )

        max_min = min( f.chemical_potencial( Vch, j ) )

        v2_equlibrium.append( otrezok( chemical_point_equlibrium, Vch2[0], Vch2[-1], f.T)[ list(f.T).index( j ) ] )

    return v_equlibrium, v2_equlibrium


def VliqVgas( vliq, V1, V2, t ): #созжание равносильных по хим. потенциалу массивов объемов
    # для нахождения значений давлений

    global otrezok

    def VDW( v, T ):

        return f.chemical_potencial( v, T ) - f.chemical_potencial( vliq[ list( vliq ).index( f.v_i ) ] , T )

    vgas = np.asarray( [ 42.0 for i in vliq ] )

    for f.v_i in vliq:

        #print( 'OK1' )

        vgas[ list(vliq).index( f.v_i ) ] = otrezok( VDW, V1[ list( f.T ).index( t ) ], V2[ list( f.T ).index( t ) ], [ t, t ] )[ 0 ]

        #print( 'OK2' )

    return vgas





