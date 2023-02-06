import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PengRobinsonF as PRf
import ch as ch
import function as f
import numpy.linalg as lin
from numba import jit

#y = PRf.Newton( PRf.y0, PRf.V0 )

v_eq1, v_eq2 = ch.equlibrium()

V1_mass = ch.critical_point( f.V[0], f.V[400], f.T )

V2_mass = ch.critical_point( f.V[440], f.V[ f.Number - 3 ], f.T )

Pliq = []

Pgas = []

plt.figure( figsize = ( 15, 7 ) )
plt.xlabel( r'$ P_{liq}, atm $', fontsize = 13 )
plt.ylabel( '$P_{gas}, atm$', fontsize = 13 )
plt.grid()

for j in f.T:

    v_liq = np.linspace( v_eq1[ list( f.T ).index( j ) ], V1_mass[ list( f.T ).index( j ) ], 100 )

    v_gas = ch.VliqVgas( v_liq[1:-1], V2_mass, v_eq2, j )

    Pliq = f.Van_der_Vaals( j, v_liq )

    Pgas = f.Van_der_Vaals( j, v_gas )

    if j == f.T[ 0 ]:

        x = Pliq[1:-31] *0.8e-5

        y = Pliq[1:-31] *0.8e-5

    plt.plot( Pliq[1:-41] *0.8e-5, np.array( [ Pgas[j] + 20.0 * 1.0e5 * 20.0 / 97 * j for j in range( len( list( Pgas ) ) ) ] )[:-40] * 0.4e-5, f.color[ list( f.T ).index( j ) ])


plt.legend( [ str(i) + ' K' for i in f.T ], fontsize = 13 )
plt.plot( x, y, 'ko-' )

plt.savefig( 'Peng_Robinson_Pgas(P_liq)_with_bisectorN2.png', dpi = 1000 )

