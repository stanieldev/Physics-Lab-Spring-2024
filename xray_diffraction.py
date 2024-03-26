# Imports
import numpy as np
import matplotlib.pyplot as plt


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


SHOW_NICKEL_FILTER = True
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
