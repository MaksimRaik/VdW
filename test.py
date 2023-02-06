import numpy as np
import matplotlib.pyplot as plt
import function as f
import ch

plt.axis( [0 , 0.00201, -0.016, 0.01] )

for i in f.T:

    plt.plot( f.V, f.chemical_potencial( f.V, i ) / 10 ** 6 )

plt.grid()
plt.show()
