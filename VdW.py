import numpy as np
import matplotlib.pyplot as plt
import function as f
import ch

# 1)файл Van_der_Waals.py оздан для построения графиков основных функций уравнений состояний
# и термодинамичесикх величин.
# 2) файл function.py содержит описание функций, графики которых строятся в файле Van_der_Waals.py
# 3) файл Van_der_Waals_2.0.py создан для нахождения равных значений химического потенциала газовой
# ижидкой фазы
# 4) ch.py содержит функции необходимые для нахождения равных значений химического потенциала для
# жидкой и газообразной фаз

#variables and list

T = np.asarray( [ 545 + 10 * i for i in range(5) ] )

V = np.asarray([f.b + i * f.eps for i in range(1, f.Number)])

color = [ 'r-', 'b-', 'g-', 'c-', 'k-', 'm-' ]

#plot graph the equation of Van der Waals

plt.figure( figsize = ( 10, 5 ) )
plt.grid()
plt.axis( [ 0, 0.0004, 0, 60 ] )
plt.xlabel( r'$ V \times mol^{-1},\ cm^3 $', fontsize = 10 )
plt.ylabel( '$P, atm$', fontsize = 10 )

for i in T:

    plt.plot( V, f.Van_der_Vaals( i, V ) / 10 ** 6 )


plt.legend( [ str(i) + ' K' for i in T ], fontsize = 13 )
plt.plot( V, f.Si( V ) / 10 ** 6 )
#plt.show()
plt.savefig('Van_der_Waals.png', dpi = 1000 )

#plot graph for free energy

plt.figure( figsize = ( 10, 5 ) )
plt.grid()
plt.axis( [ 0, 0.0004, -15.0, 12.0 ] )
plt.xlabel( r'$ V \times mol^{-1},\ cm^3 $', fontsize = 13 )
plt.ylabel( r'$F, MJ$', fontsize = 13 )

for i in T:

    plt.plot( V, f.free_energy( i, V ) / 10 ** 3 )

plt.legend( [ str( i ) + ' K' for i in T ], fontsize = 13 )
#plt.show()
plt.savefig('free_energy_all.png', dpi = 400 )

#chemical potencial

V1_mass = ch.critical_point( V[0], V[400], T )

V2_mass = ch.critical_point( V[440], V[ f.Number - 3 ], T )

plt.figure( figsize = ( 10, 4 ) )
plt.grid( )
plt.axis( [0 , 0.00201, -0.02, 0.01] )
plt.xlabel( r'$ V \times mol^{-1},\ cm^3 $', fontsize = 10 )
plt.ylabel( r'$\mu, MJ$', fontsize = 10 )

for j in T:

    Vch = np.linspace( V[0], V1_mass[ list(T).index( j ) ], 2000 )

    Vch2 = np.linspace( V2_mass[ list(T).index( j ) ], 0.002, 2000 )

    plt.plot( Vch, f.chemical_potencial( Vch, j ) / 10 ** 6, color[ list(T).index( j ) ] )
    plt.plot( Vch2, f.chemical_potencial( Vch2, j ) / 10 ** 6, color[ list(T).index( j ) ] )

plt.legend( [ str( j ) + ' K' for j in T ], fontsize = 13 )
plt.savefig('chemical_potencial_equalibrim ( -0.05, 0.01 ).png', dpi = 1000 )
#plt.show()
