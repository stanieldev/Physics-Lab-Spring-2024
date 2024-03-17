# Imports
import numpy as np
import matplotlib.pyplot as plt





SHOW_DATASET = True
if SHOW_DATASET:

    # Load the data
    data_15kV = np.loadtxt('xray_diffraction/15KV.txt', delimiter='\t', skiprows=3)
    data_20kV = np.loadtxt('xray_diffraction/20KV.txt', delimiter='\t', skiprows=3)
    data_25kV = np.loadtxt('xray_diffraction/25KV.txt', delimiter='\t', skiprows=3)
    data_30kV = np.loadtxt('xray_diffraction/30KV.txt', delimiter='\t', skiprows=3)
    data_35kV = np.loadtxt('xray_diffraction/35KV.txt', delimiter='\t', skiprows=3)

    # Plot the data
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





SHOW_NORMALIZED_DATASET = True
if SHOW_NORMALIZED_DATASET:

    # Load the data
    data_15kV = np.loadtxt('xray_diffraction/15KV.txt', delimiter='\t', skiprows=3)
    data_20kV = np.loadtxt('xray_diffraction/20KV.txt', delimiter='\t', skiprows=3)
    data_25kV = np.loadtxt('xray_diffraction/25KV.txt', delimiter='\t', skiprows=3)
    data_30kV = np.loadtxt('xray_diffraction/30KV.txt', delimiter='\t', skiprows=3)
    data_35kV = np.loadtxt('xray_diffraction/35KV.txt', delimiter='\t', skiprows=3)

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


