# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
from lmfit import Model
MU_0 = 4*np.pi*1e-7



def load_field_current_dependence():

    # Load the data
    raw_data = np.loadtxt('polarization_rotation/field_current_dependence.tsv', delimiter='\t', skiprows=1)
    
    # Extract the columns
    current = raw_data[:, 0]
    field = raw_data[:, 1]

    # Find the 0s in the field
    V0_negative = field[8]
    V0_positive = field[9]

    # Subtract the 0s from the field
    field[:9] -= V0_negative
    field[9:] -= V0_positive
    
    # Return the data
    return current, field


CURRENT_DEPENDENCE = False
if CURRENT_DEPENDENCE:

    # Load data
    current, field = load_field_current_dependence()

    # Plot with the data
    plt.plot(current, field, 'o')
    plt.xlabel('Current (A)')
    plt.ylabel('Field (mV)')
    plt.title('Magnetic Field vs Current')

    # Create a model
    model = LinearModel()
    params = model.make_params()
    result = model.fit(field, params, x=current)
    print(result.fit_report())

    # Find the slope
    slope = result.best_values['slope']
    slope_err = result.params['slope'].stderr
    expected = MU_0 * (1400/0.15) * 1e4
    CV = expected / slope
    dCV = CV * np.sqrt((slope_err/slope)**2)
    print(CV)
    print(dCV)

    # Plot the fit
    plt.plot(current, result.best_fit, label='Fit')
    plt.legend()
    plt.show()


def magnetic_field_fit(z, m, L, R):
    # m is the packing constant ~< 10
    # L is the length of the solenoid ~ 15cm
    # R is the average radius of the solenoid ~ 1.3cm

    # Constants
    I = 1.0      # A
    D = 1.02e-3  # m
    Gauss_to_mV = 1/1.1158575212250224  # mV/Gauss
    Tesla_to_Gauss = 1e4

    # Calculate the magnetic field
    K = m * I * MU_0 / (2 * D)
    K *= Gauss_to_mV * Tesla_to_Gauss
    seg1 = (L/2 - z) / np.sqrt((L/2 - z)**2 + R**2)
    seg2 = (-L/2 - z) / np.sqrt((-L/2 - z)**2 + R**2)

    # Return the magnetic field in mV
    return K * (seg1 - seg2)


Z_DEPENDENCE = False
if Z_DEPENDENCE:

    # Load the data
    raw_data = np.loadtxt('polarization_rotation/field_coil_variance.tsv', delimiter='\t', skiprows=1)
    
    # Extract the columns
    z = raw_data[:, 0] - 7.5
    field = (raw_data[:, 1] + raw_data[:, 2])/2

    # Plot with the data
    plt.plot(z, field, 'o')
    plt.xlabel('z (cm)')
    plt.ylabel('Field (mV)')
    plt.title('Magnetic Field vs z')

    # Create a model
    model = Model(magnetic_field_fit)
    params = model.make_params(m=5, L=15, R=1.3)
    result = model.fit(field, params, z=z)
    print(result.fit_report())

    # Use the regression as a function to integrate
    def B(z):
        return result.eval(z=z)
    area = np.trapz([B(z) for z in np.linspace(-5, 5, 100)], dx=0.1)
    print(f"Average over rod: {area/(5-(-5))}")
    print(f"Peak at middle: {B(0)}")

    # Plot the fit
    X = np.linspace(-8, 8, 100)
    plt.plot(X, result.eval(z=X), label='Fit')

    # Plot vertical lines at +/- 5cm to show the rod
    plt.axvline(-5, color='black', linestyle='--')
    plt.axvline(5, color='black', linestyle='--')

    # Plot a line at the average field over the rod
    plt.axhline(area/(5-(-5)), color='red', linestyle='--', label='Average over rod')

    # Show the area under the curve from -5 to 5
    Y = np.linspace(-5, 5, 100)
    plt.fill_between(Y, result.eval(z=Y), alpha=0.2)

    # Show the plot
    # force y-axis to be 0 at the bottom
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


DC_POLARIZATION = False
if DC_POLARIZATION:
    
    # Load the data
    raw_data = np.loadtxt('polarization_rotation/dc_polarization.tsv', delimiter='\t', skiprows=1)
    theta_err = 0.25  # degrees
    
    # Extract the columns (angle in degrees)
    angle = raw_data[:, 0]
    voltage = raw_data[:, 1]
    angle_rad = np.radians(angle)

    # Plot with the data
    plt.errorbar(angle, voltage, xerr=theta_err, fmt='o')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Voltage (V)')
    plt.title('DC Polarization Rotation')

    # Find a cos² fit
    model = Model(lambda x, A, B, p: A * np.cos(x + p)**2 + B)
    params = model.make_params(A=1, B=0, p=0)
    result = model.fit(voltage, params, x=angle_rad)
    print(result.fit_report())
    print(np.degrees(result.best_values['p']), np.degrees(result.params['p'].stderr))

    # Plot the fit
    X = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.degrees(X), result.eval(x=X), label='Fit')
    plt.legend()
    plt.show()


DC_POLARIZATION_2 = True
if DC_POLARIZATION_2:

    # Load the data
    raw_data = np.loadtxt('polarization_rotation/dc_polarization_2.tsv', delimiter='\t', skiprows=1)
    theta_err = 0.25  # degrees
    C = 0.9838
    
    # Extract the columns (angle in degrees)
    current = raw_data[:, 0]
    initial = raw_data[:, 1]
    final = raw_data[:, 2]

    # Find new lists
    dAngle = final - initial
    field_Gauss = 1.116 * 105.108514 * current * C

    # Plot with the data
    plt.errorbar(field_Gauss, dAngle, fmt='o', label='Data', xerr=0.01, yerr=2*theta_err)
    plt.xlabel('Magnetic Field (Gauss)')
    plt.ylabel('Angle (degrees)')
    plt.title('DC Polarization Rotation')

    # Find a linear fit
    model = LinearModel()
    params = model.make_params()
    result = model.fit(dAngle, params, x=field_Gauss, weights=1/(2*theta_err)**2)
    print(result.fit_report())

    # Plot the fit
    plt.plot(field_Gauss, result.best_fit, label='Fit')
    plt.legend()
    plt.show()


