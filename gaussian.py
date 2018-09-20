# simple script to fit a Gaussian to fake data

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Gaussian function
def gaussian(x, a, b, c, d):
    val = d + (a * np.exp(-(x - b)**2 / c**2))
    return val

# number of iterations
nit=1000

# array to store the result of each iteration
result = np.zeros(nit)

# fake x-data
x = np.arange(0.1,10.,0.1)

plt.ion()
plt.clf()

for i in range(0, nit):

    # fake y-data
	y = gaussian(x,5.,5.,1.,2.)+np.random.normal(1.,0.2,x.size)

    # fit the data
	popt, pcov = curve_fit(gaussian, x, y, p0=[5.,5.,1.,3.])

    # fitted model
	z=gaussian(x,popt[0],popt[1],popt[2],popt[3])

    # save the fitted center of the gaussian
	result[i] = popt[1]
	
	plt.plot(x,y,'or')
	plt.plot(x,z)
	plt.draw()
	raw_input(':')
	plt.clf()
	
# plot a histogram of fitted centers of the gaussian
plt.hist(result,bins=20)

