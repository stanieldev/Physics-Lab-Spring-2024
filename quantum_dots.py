# Imports
import numpy as np
import matplotlib.pyplot as plt


# Materials
class CdSe:
    def __init__(self) -> None:
        self.E_g = 1.74  # eV
        self.m_e = 0.13  # Proportion of electron mass
        self.m_h = 0.45  # Proportion of electron mass

class InP :
    def __init__(self) -> None:
        self.E_g = 1.34  # eV
        self.m_e = 0.08  # Proportion of electron mass
        self.m_h = 0.60  # Proportion of electron mass



# OneDimSquareWell
def E1(diameter_nm: float, material=None):
    k = 0.376125244618  # eV nm^2  ( hc²/(8 m_ec²) )
    return k / (diameter_nm ** 2)

# ThreeDimSquareWell
def E2(diameter_nm: float, material=None):
    k = 1.12837573386  # eV nm^2  ( 3 hc²/(8 m_ec²) )
    return k / (diameter_nm ** 2)

# ThreeDimSquareWellGapped
def E3(diameter_nm: float, material: CdSe or InP):
    return material.E_g + E2(diameter_nm)

# SphericalWellGapped
def E4(diameter_nm: float, material: CdSe or InP):
    k = 0.376125244618  # eV nm^2  ( hc²/(8 m_ec²) )
    return k / (diameter_nm ** 2) + material.E_g

# SphericalWellGappedReduced
def E5(diameter_nm: float, material: CdSe or InP):
    k = 0.376125244618  # eV nm^2  ( hc²/(8 m_ec²) )
    return (k / (diameter_nm ** 2))*(1/material.m_e + 1/material.m_h) + material.E_g




# Energy to Wavelength
def Lambda(E: float):
    return 1240/E  # nm









# Data Constants
CdSe_diameters = [2.3, 2.6, 3.3, 4.6, 5.6, 6.9]  # nm
CdSe_wavelengths = [524.3, 535.3, 558.3, 620.7, 635.4, 656.7]  # nm
CdSe_HWHM = [26.94, 17.73, 15.62, 10.43, 18.58, 16.11]  # nm

InP_diameters = np.array([4.7, 5.1, 5.4, 5.8])  # nm
InP_wavelengths = np.array([528.7, 568.2, 596.9, 621.2])  # nm
InP_HWHM = np.array([24.96, 17.33, 23.71, 30.08])  # nm


# Chi-Squared Function
def chi_squared(model, diameters, wavelengths, HWHM):
    chi_squared = 0
    for d, w, H in zip(diameters, wavelengths, HWHM):
        chi_squared += (Lambda(model(d, CdSe())) - w)**2 / (H**2)
    return chi_squared

# # Print Chi-Squared Values
# print("Chi-Squared Values:")
# DIVISOR = 5
# print("1D Square Well: ", chi_squared(E1, CdSe_diameters, CdSe_wavelengths, CdSe_HWHM)/DIVISOR)
# print("3D Square Well: ", chi_squared(E2, CdSe_diameters, CdSe_wavelengths, CdSe_HWHM)/DIVISOR)
# print("3D Square Well with Gap: ", chi_squared(E3, CdSe_diameters, CdSe_wavelengths, CdSe_HWHM)/DIVISOR)
# print("Spherical Well with Gap: ", chi_squared(E4, CdSe_diameters, CdSe_wavelengths, CdSe_HWHM)/DIVISOR)
# print("Spherical Well with Gap and Reduced Mass: ", chi_squared(E5, CdSe_diameters, CdSe_wavelengths, CdSe_HWHM)/DIVISOR)




# # Plot InP wavelength vs diameter
# plt.figure()
# plt.xlabel('Diameter (nm)')
# plt.ylabel('Peak Wavelength (nm)')
# plt.title('InP Peak Wavelength vs Diameter')
# plt.plot(InP_diameters, InP_wavelengths, 'o')

# # Plot the 5 models for InP
# diameters = np.linspace(4.5, 6, 1000)
# plt.plot(diameters, [Lambda(E5(d, InP())) for d in diameters], label='Spherical Well with Gap and Reduced Mass')
# plt.legend()

# # Show plot
# plt.show()


# Plot an inverse plot of InP wavelength vs diameter
plt.figure()
plt.xlabel('Peak Wavelength (nm)')
plt.ylabel('Diameter (nm)')
plt.title('InP Diameter vs. Peak Wavelength')
plt.plot(InP_wavelengths, InP_diameters, 'o', color="blue")
plt.plot(InP_wavelengths + InP_HWHM, InP_diameters, 'o', alpha=0.75, color="blue")
plt.plot(InP_wavelengths + 2*InP_HWHM, InP_diameters, 'o', alpha=0.50, color="blue")
plt.plot(InP_wavelengths + 3*InP_HWHM, InP_diameters, 'o', alpha=0.25, color="blue")
plt.plot(InP_wavelengths - InP_HWHM, InP_diameters, 'o', alpha=0.75, color="blue")
plt.plot(InP_wavelengths - 2*InP_HWHM, InP_diameters, 'o', alpha=0.50, color="blue")
plt.plot(InP_wavelengths - 3*InP_HWHM, InP_diameters, 'o', alpha=0.25, color="blue")

# Inverse SphericalWellGappedReduced
def inv_E5(energy: float, material: CdSe or InP):
    k = 0.376125244618  # eV nm^2  ( hc²/(8 m_ec²) )
    return np.sqrt( k*(1/material.m_e + 1/material.m_h) / (energy + material.E_g) )


# Plot inv5E
wavelength = np.linspace(500, 650, 1000)
plt.plot(wavelength, [inv_E5(1240/w, InP()) for w in wavelength], label='Inverse Spherical Well with Gap and Reduced Mass')
plt.show()


# Plot CdSe wavelength vs diameter
plt.figure()
plt.xlabel('Diameter (nm)')
plt.ylabel('Peak Wavelength (nm)')
plt.title('CdSe Peak Wavelength vs Diameter')
plt.plot(CdSe_diameters, CdSe_wavelengths, 'o')

# Plot the 5 models for CdSe
diameters = np.linspace(2, 7, 1000)
# plt.plot(diameters, [Lambda(E1(d)) for d in diameters], label='1D Square Well')
# plt.plot(diameters, [Lambda(E2(d)) for d in diameters], label='3D Square Well')
plt.plot(diameters, [Lambda(E3(d, CdSe())) for d in diameters], label='3D Square Well with Gap')
plt.plot(diameters, [Lambda(E4(d, CdSe())) for d in diameters], label='Spherical Well with Gap')
plt.plot(diameters, [Lambda(E5(d, CdSe())) for d in diameters], label='Spherical Well with Gap and Reduced Mass')
plt.legend()

# Show plot
plt.show()




# # Plot CdSe wavelength vs diameter
# plt.figure()
# plt.xlabel('Diameter (nm)')
# plt.ylabel('log(Peak Wavelength) (nm)')
# plt.title('CdSe Peak Wavelength vs Diameter')
# plt.plot(CdSe_diameters, np.log(CdSe_wavelengths), 'o')

# # Plot the 5 models for CdSe
# diameters = np.linspace(2, 7, 1000)
# plt.plot(diameters, [np.log(Lambda(E1(d))) for d in diameters], label='1D Square Well')
# plt.plot(diameters, [np.log(Lambda(E2(d))) for d in diameters], label='3D Square Well')
# plt.plot(diameters, [np.log(Lambda(E3(d, CdSe()))) for d in diameters], label='3D Square Well with Gap')
# plt.plot(diameters, [np.log(Lambda(E4(d, CdSe()))) for d in diameters], label='Spherical Well with Gap')
# plt.plot(diameters, [np.log(Lambda(E5(d, CdSe()))) for d in diameters], label='Spherical Well with Gap and Reduced Mass')
# plt.legend()

# # Show plot
# plt.show()

