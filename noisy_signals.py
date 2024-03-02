# Import relevent libraries
import numpy as np
import matplotlib.pyplot as plt

# Open the noise_data.csv file
data = np.genfromtxt('noisy_signals/noise_data.csv', delimiter=',')

# Extract the resistance and voltage squared data
resistance = data[:,0]
voltage_squared = data[:,1]
voltage_squared_error = data[:,2]

# Plot the resistance vs voltage squared
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Input Resistance ($\Omega$)')
plt.ylabel(r'$\langle V_J^2+V_N^2\rangle$ ($10^{-12} $V$^2$)')
plt.title(r'$R_{\mathrm{in}}$ vs $\langle V_J^2+V_N^2\rangle$')
OFFSET = 7.4510240812
plt.errorbar(resistance, abs(voltage_squared - OFFSET), yerr=voltage_squared_error, fmt='x')
plt.show()


NEW_DATA = voltage_squared - OFFSET

# Find the gradient of the last 2 points
m1 = (np.log(NEW_DATA[5]) - np.log(NEW_DATA[4])) / (np.log(resistance[5]) - np.log(resistance[4]))
m2 = (np.log(NEW_DATA[4]) - np.log(NEW_DATA[3])) / (np.log(resistance[4]) - np.log(resistance[3]))
m3 = (np.log(NEW_DATA[5]) - np.log(NEW_DATA[3])) / (np.log(resistance[5]) - np.log(resistance[3]))

# Print the average and standard deviation of the gradients
print('m1 = {:.10f}'.format(m1))
print('m2 = {:.10f}'.format(m2))
print('m3 = {:.10f}'.format(m3))
print('Average gradient = {:.10f}'.format((m1 + m2 + m3) / 3))
print('Standard deviation = {:.10f}'.format(np.std([m1, m2, m3])))




# # use scipy to fit a line to the data
# def linear(x, m, c):
#     return m*x + c


# for i, j in zip(resistance, voltage_squared):
#     print(i, j-OFFSET)


# LMFIT failed to find a fit, so I used scipy's curve_fit instead
# from scipy.optimize import curve_fit
# popt, pcov = curve_fit(linear, resistance, voltage_squared, sigma=voltage_squared_error)
# m, c = popt
# print('Gradient = {:.10f} ± {:.10f}'.format(m, np.sqrt(pcov[0,0])))
# print('y-intercept = {:.10f} ± {:.10f}'.format(c, np.sqrt(pcov[1,1])))

# # Plot the fit
# x = np.linspace(0, 100000, 1000)
# plt.plot(x, linear(x, m, c), label='Fit: y = {:.5f}x + {:.5f}'.format(m, c))
# plt.legend()
