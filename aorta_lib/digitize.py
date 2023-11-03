import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

nbins = 31

m = pv.read('./metrics.vtp')
long = m['Abscissas']
tawss = m['tawss']
bins = np.linspace(long.min(), long.max(), nbins)  # Creates 10 bins from 0 to 1

# Find the indices of the bins to which each value in long belongs
long_to_bin = np.digitize(long, bins) - 1
# Compute the sum of Y values for each bin
sums = np.bincount(long_to_bin, weights=tawss, minlength=len(bins)-1)
# Compute the count of Y values for each bin
counts = np.bincount(long_to_bin, minlength=len(bins)-1)

# Calculate the average of Y values for each bin
# Avoid division by zero by using where to only divide where counts is non-zero
averages = np.divide(sums, counts, where=counts>0)
print(averages)
#plt.plot(averages)
#plt.show()
