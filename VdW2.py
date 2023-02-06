import numpy as np
import matplotlib.pyplot as plt
import function as f
import ch

# продолжение расчетов для фазового равновесия газа Ван дер Ваальса
# 1) нахожение области с равными химическими потенциалами
# 2) построение графика в кооординатах давлений
#print( 'равновестные значения V для жидкой и газообразной фаз', ch.equlibrium() )

v_eq1, v_eq2 = ch.equlibrium()

# значения равных химпотенциалов лежат в промежутке ( v_eq1[j], v_eq2[j] ) для j = 1...6
#  1) из равновестных давлений восстановим значения P.
#  2) построить кривую Pliq( Pgas )

V1_mass = ch.critical_point( f.V[0], f.V[400], f.T )

V2_mass = ch.critical_point( f.V[440], f.V[ f.Number - 3 ], f.T )

Pliq = []

Pgas = []

plt.figure( figsize = ( 11, 10 ) )
plt.xlabel( r'$ P_{liq}, atm $', fontsize = 13 )
plt.ylabel( '$P_{gas}, atm$', fontsize = 13 )
plt.grid()

for j in f.T:

    v_liq = np.linspace( v_eq1[ list( f.T ).index( j ) ], V1_mass[ list( f.T ).index( j ) ], 100 )

    v_gas = ch.VliqVgas( v_liq[1:-1], V2_mass, v_eq2, j )

    Pliq = f.Van_der_Vaals( j, v_liq )

    Pgas = f.Van_der_Vaals( j, v_gas )

    if j == f.T[ 0 ]:

        x = Pliq

        y = Pliq

    plt.plot( Pliq[1:-1] / ( 10 ** 6 ), Pgas / ( 10 ** 6 ), f.color[ list( f.T ).index( j ) ] )

plt.legend( [ str(i) + ' K' for i in f.T ], fontsize = 13 )
plt.plot( x / ( 10 ** 6 ), y / ( 10 ** 6 ), 'k-' )

plt.savefig( 'Pgas(P_liq)_with_bisector2.0.png', dpi = 1000 )

#################################################################################
#################################################################################

plt.figure( figsize = ( 11, 10 ) )
plt.xlabel( r'$ V \times mol^{-1},\ cm^3 $', fontsize = 13 )
plt.ylabel( '$P, atm$', fontsize = 13 )
plt.grid()

for j in f.T:

    v_liq = np.linspace( v_eq1[ list( f.T ).index( j ) ], V1_mass[ list( f.T ).index( j ) ], 100 )

    v_gas = ch.VliqVgas( v_liq[1:-1], V2_mass, v_eq2, j )

    Pliq = f.Van_der_Vaals( j, v_liq )

    Pgas = f.Van_der_Vaals( j, v_gas )

    plt.plot( v_liq, Pliq / ( 10 ** 6 ), f.color[ list( f.T ).index( j ) ], label = str( f.T[ list( f.T ).index( j ) ] ) + ' K' )
    plt.plot( v_gas, Pgas / ( 10 ** 6 ), f.color[ list( f.T ).index( j ) ] )

    #plt.plot( Pliq[1:-1] / ( 10 ** 6 ), Pgas / ( 10 ** 6 ), f.color[ list( f.T ).index( j ) ] )

plt.legend( fontsize = 13 )

plt.savefig( 'VdW2.0.png', dpi = 1000 )





