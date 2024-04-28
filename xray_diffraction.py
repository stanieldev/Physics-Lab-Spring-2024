# Imports
import numpy as np
import matplotlib.pyplot as plt
import lmfit


SHOW_DATASET = False
if SHOW_DATASET:

    # Load the data
    data_8kV = np.loadtxt('xray_diffraction/8KV.txt', delimiter='\t', skiprows=3)
    data_10kV = np.loadtxt('xray_diffraction/10KV.txt', delimiter='\t', skiprows=3)
    data_12kV = np.loadtxt('xray_diffraction/12KV.txt', delimiter='\t', skiprows=3)
    data_15kV = np.loadtxt('xray_diffraction/15KV.txt', delimiter='\t', skiprows=3)
    data_20kV = np.loadtxt('xray_diffraction/20KV.txt', delimiter='\t', skiprows=3)
    data_25kV = np.loadtxt('xray_diffraction/25KV.txt', delimiter='\t', skiprows=3)
    data_30kV = np.loadtxt('xray_diffraction/30KV.txt', delimiter='\t', skiprows=3)
    data_35kV = np.loadtxt('xray_diffraction/35KV_old.txt', delimiter='\t', skiprows=3)

    # Plot the data
    plt.plot(data_8kV[:, 0], data_8kV[:, 1], label='8 kV')
    plt.plot(data_10kV[:, 0], data_10kV[:, 1], label='10 kV')
    plt.plot(data_12kV[:, 0], data_12kV[:, 1], label='12 kV')
    plt.plot(data_15kV[:, 0], data_15kV[:, 1], label='15 kV')
    plt.plot(data_20kV[:, 0], data_20kV[:, 1], label='20 kV')
    plt.plot(data_25kV[:, 0], data_25kV[:, 1], label='25 kV')
    plt.plot(data_30kV[:, 0], data_30kV[:, 1], label='30 kV')
    plt.plot(data_35kV[:, 0], data_35kV[:, 1], label='35 kV')
    
    # Set up the plot
    plt.title('X-ray Diffraction : Intensity vs. Crystal Angle')
    plt.xlabel('Crystal Angle θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.show()


SHOW_NORMALIZED_DATASET = False
if SHOW_NORMALIZED_DATASET:

    # Load the data
    data_15kV = np.loadtxt('xray_diffraction/15KV.txt', delimiter='\t', skiprows=3)
    data_20kV = np.loadtxt('xray_diffraction/20KV.txt', delimiter='\t', skiprows=3)
    data_25kV = np.loadtxt('xray_diffraction/25KV.txt', delimiter='\t', skiprows=3)
    data_30kV = np.loadtxt('xray_diffraction/30KV.txt', delimiter='\t', skiprows=3)
    data_35kV = np.loadtxt('xray_diffraction/35KV_old.txt', delimiter='\t', skiprows=3)

    # Plot the data
    plt.plot(data_15kV[:, 0], data_15kV[:, 1]/(15-11.588), label='15 kV')
    plt.plot(data_20kV[:, 0], data_20kV[:, 1]/(20-11.588), label='20 kV')
    plt.plot(data_25kV[:, 0], data_25kV[:, 1]/(25-11.588), label='25 kV')
    plt.plot(data_30kV[:, 0], data_30kV[:, 1]/(30-11.588), label='30 kV')
    plt.plot(data_35kV[:, 0], data_35kV[:, 1]/(35-11.588), label='35 kV')
    
    # Set up the plot
    plt.title('X-ray Diffraction : Intensity vs. Crystal Angle')
    plt.xlabel('Crystal Angle θ (degrees)')
    plt.ylabel('Intensity / kV (a.u./kV)')
    plt.legend()
    plt.show()


SHOW_NICKEL_FILTER = False
if SHOW_NICKEL_FILTER:

    # Load the data
    data_35kV = np.loadtxt('xray_diffraction/35KV.txt', delimiter='\t', skiprows=3)
    data_35kV_ni = np.loadtxt('xray_diffraction/35KV_Ni.txt', delimiter='\t', skiprows=3)

    # Plot the data
    plt.plot(data_35kV[:, 0], data_35kV[:, 1], label='Unfiltered')
    plt.plot(data_35kV_ni[:, 0], data_35kV_ni[:, 1], label='Ni Filter')

    # Set up the plot
    plt.title('X-ray Diffraction : Intensity vs. Crystal Angle')
    plt.xlabel('Crystal Angle θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.show()


PLOT_KRAMERS_LAW = False
if PLOT_KRAMERS_LAW:
    
    # Load the data
    data_35kV = np.loadtxt('xray_diffraction/kramers_law_nickel.txt', delimiter='\t', skiprows=3)

    # Unpack the data
    theta = data_35kV[:, 0]
    intensity = data_35kV[:, 1]

    # Set up the graph
    plt.title('X-ray Diffraction : Intensity vs. Crystal Angle')
    plt.xlabel('Crystal Angle θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.xlim(5, 25)
    plt.ylim(0, 7000)

    # Find a gaussian between 21 and 24 degrees
    theta_subset = theta[(theta >= 21) & (theta <= 24)]
    intensity_subset = intensity[(theta >= 21) & (theta <= 24)]
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x-mu)**2 / sigma**2)
    model = lmfit.Model(gaussian)
    params = model.make_params(A=6000, mu=22.5, sigma=1)
    result = model.fit(intensity_subset, params, x=theta_subset)
    print(result.fit_report())
    
    # Plot the data
    plt.plot(theta, intensity, label='Unfiltered')
    plt.plot(theta_subset, result.best_fit, label='Gaussian Fit')

    # Show the plot
    plt.show()




THEORETICAL_KRAMERS_LAW = True
if THEORETICAL_KRAMERS_LAW:

    # Load the data
    data_35kV = np.loadtxt('xray_diffraction/kramers_law_nickel.txt', delimiter='\t', skiprows=3)

    # Unpack the data
    theta = data_35kV[:, 0]
    intensity = data_35kV[:, 1]

    # Set up the graph
    plt.title('X-ray Diffraction : Intensity vs. Crystal Angle')
    plt.xlabel('Crystal Angle θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.xlim(5, 25)
    plt.ylim(0, 7000)

    # Use Kramers Law to find the intensity
    V = 35e+3   # Voltage in Volts
    A = 0.4026  # nm

    def _wavelength_nm(a=A):
        THETA = 22.53*(np.pi/180)  # rad
        return a*np.sin(THETA)

    def _intensity(theta_rad, K, a=A):
        COEFF = (V/1240)*a
        return (K/a**2) * (COEFF * np.sin(theta_rad) - 1) / (np.sin(theta_rad)**2)

    # Regression
    theta_subset = theta[(theta >= 21) & (theta <= 24)]
    intensity_subset = intensity[(theta >= 21) & (theta <= 24)]
    model = lmfit.Model(_intensity)
    params = model.make_params(K=6000)
    result = model.fit(intensity_subset, params, theta_rad=theta_subset*(np.pi/180))
    print(result.fit_report())

    # Plot the data
    plt.plot(theta, intensity, label='Unfiltered')
    plt.plot(theta_subset, result.best_fit, label='Kramers Law Fit')

    # Show the plot
    plt.legend()
    plt.show()


