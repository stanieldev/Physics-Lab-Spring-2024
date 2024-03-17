# Import relevent libraries
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm



# SCA Control panel
PLOT_MANUAL_CAESIUM = False
ENERGY_MANUAL_CAESIUM = False
MORE_MANUAL_CAESIUM = False

# MCA Control panel
PLOT_MCA_Cs137 = True
PLOT_MCA_Mn54 = False
PLOT_MCA_Ba133 = False
GLOBALMIN = 100  # keV






# Find Gaussian function
def find_gaussian(x, y, MIN=None, MAX=None):

    # Initialize default values
    if MIN is None: MIN = min(x)
    if MAX is None: MAX = max(x)

    # Filter data
    x_data = x[(x > MIN) & (x < MAX)]
    y_data = y[(x > MIN) & (x < MAX)]

    # Find a normal distribution for the data using LMFIT
    model = lm.models.GaussianModel()
    params = model.guess(y_data, x=x_data)
    result = model.fit(y_data, params, x=x_data)
    return result

# Find Offset Gaussian function
def find_offset_gaussian(x, y, MIN=None, MAX=None):
    
        # Initialize default values
        if MIN is None: MIN = min(x)
        if MAX is None: MAX = max(x)
    
        # Filter data
        x_data = x[(x > MIN) & (x < MAX)]
        y_data = y[(x > MIN) & (x < MAX)]
    
        # Find a normal distribution for the data using LMFIT
        def offset_gaussian(x, amplitude, center, sigma, voffset):
            return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2) + voffset
        model = lm.Model(offset_gaussian)
        params = model.make_params(amplitude=100, center=200, sigma=0.1, voffset=0)
        result = model.fit(y_data, params, x=x_data)
        return result

# Find Sigmoid function
def find_sigmoid(x, y, MIN=None, MAX=None):

    # Initialize default values
    if MIN is None: MIN = min(x)
    if MAX is None: MAX = max(x)

    # Filter data
    x_data = x[(x > MIN) & (x < MAX)]
    y_data = y[(x > MIN) & (x < MAX)]

    # Find a sigmoid function for the data using LMFIT
    def sigmoid(x, amplitude, steepness, center, voffset):
        return amplitude / (1 + np.exp(-steepness * (x - center))) + voffset
    model = lm.Model(sigmoid)
    params = model.make_params(amplitude=1300, steepness=1, center=400, voffset=400)
    result = model.fit(y_data, params, x=x_data)
    return result





# MCA Caesium data
if PLOT_MCA_Cs137:

    # Load MCA data
    mca_data = np.loadtxt('gamma_spectroscopy/caesium/caesium.tsv', delimiter='\t', skiprows=25)
    energy_kev = mca_data[:, 1][mca_data[:, 1] > GLOBALMIN]
    counts = mca_data[:, 2][mca_data[:, 1] > GLOBALMIN]

    # Plot MCA data
    plt.plot(energy_kev, counts, label='Measured Data', color='black')
    plt.xlabel(r'Energy ($keV$)')
    plt.ylabel(r'Counts')
    plt.title(r'MCA $^{137}$Cs : Counts vs Energy')
    plt.xlim(GLOBALMIN, max(energy_kev))
    plt.xticks(np.arange(GLOBALMIN, max(energy_kev), 100))

    # Regressions
    SHOW_REGRESSIONS = True
    if SHOW_REGRESSIONS:

        # Photopeak
        MIN, MAX = 600, 800
        photopeak_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(photopeak_result.fit_report())

        # Compton edge
        MIN, MAX = 350, 600
        compton_result = find_sigmoid(energy_kev, counts, MIN, MAX)
        print(compton_result.fit_report())

        # Backscatter peak
        MIN, MAX = 175, 300
        backscatter_result = find_offset_gaussian(energy_kev, counts, MIN, MAX)
        print(backscatter_result.fit_report())

        # Plot the fits
        PHOT_X = np.linspace(min(energy_kev), max(energy_kev), 1000)
        BACK_MAX = 2*backscatter_result.params['center'].value - GLOBALMIN
        COMP_MAX = 2*compton_result.params['center'].value - BACK_MAX
        BACK_X = np.linspace(0, BACK_MAX, 100)
        COMP_X = np.linspace(BACK_MAX, COMP_MAX, 100)

        plt.plot(PHOT_X, photopeak_result.eval(x=PHOT_X), label='Photopeak Fit', color='red')
        plt.plot(COMP_X, compton_result.eval(x=COMP_X), label='Compton Edge Fit', color='green')
        plt.plot(BACK_X, backscatter_result.eval(x=BACK_X), label='Backscatter Peak Fit', color='blue')
        

    # Show plot
    plt.legend()
    plt.show()

# MCA Manganese data
if PLOT_MCA_Mn54:

    # Load MCA data
    mca_data = np.loadtxt('gamma_spectroscopy/manganese/manganese.tsv', delimiter='\t', skiprows=25)
    energy_kev = mca_data[:, 1][mca_data[:, 1] > GLOBALMIN]
    counts = mca_data[:, 2][mca_data[:, 1] > GLOBALMIN]

    # Plot MCA data
    plt.plot(energy_kev, counts, label='Measured Data', color='black')
    plt.xlabel(r'Energy ($keV$)')
    plt.ylabel(r'Counts')
    plt.title(r'MCA $^{54}$Mn : Counts vs Energy')
    plt.xlim(GLOBALMIN, max(energy_kev))
    plt.xticks(np.arange(GLOBALMIN, max(energy_kev), 100))

    # Regressions
    SHOW_REGRESSIONS = False
    if SHOW_REGRESSIONS:

        # Photopeak
        MIN, MAX = 625, 750
        photopeak_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(photopeak_result.fit_report())

        # Compton edge
        MIN, MAX = 300, 550
        compton_result = find_sigmoid(energy_kev, counts, MIN, MAX)
        print(compton_result.fit_report())

        # Backscatter peak
        MIN, MAX = 166, 240
        backscatter_result = find_offset_gaussian(energy_kev, counts, MIN, MAX)
        print(backscatter_result.fit_report())

        # Plot the fits
        PHOT_X = np.linspace(min(energy_kev), max(energy_kev), 1000)
        BACK_MAX = 2*backscatter_result.params['center'].value - GLOBALMIN
        COMP_MAX = 2*compton_result.params['center'].value - BACK_MAX
        BACK_X = np.linspace(0, BACK_MAX, 100)
        COMP_X = np.linspace(BACK_MAX, COMP_MAX, 100)

        plt.plot(PHOT_X, photopeak_result.eval(x=PHOT_X), label='Photopeak Fit', color='red')
        plt.plot(COMP_X, compton_result.eval(x=COMP_X), label='Compton Edge Fit', color='green')
        plt.plot(BACK_X, backscatter_result.eval(x=BACK_X), label='Backscatter Peak Fit', color='blue')

    # Show plot
    plt.legend()
    plt.show()











# Manual caesium data
if PLOT_MANUAL_CAESIUM:

    # Load manual caesium data
    caesium_data = np.loadtxt('gamma_spectroscopy/manual_caesium.tsv', delimiter='\t', skiprows=1)
    voltage, counts = caesium_data.T

    # Plot manual caesium data
    plt.plot(voltage, counts, label='Measured Data', marker='o', linestyle='None', color='black')
    plt.xlabel(r'Lower Voltage ($V$)')
    plt.ylabel(r'Counts')
    plt.title(r'Manual $^{137}$Cs : Counts vs Lower Voltage')

    # Pre-fit parameters
    MIN, MAX = 5.5, 8
    PHOT_X = np.linspace(1, 10, 1000)
    x_data = voltage[(voltage > MIN) & (voltage < MAX)]
    y_data = counts[(voltage > MIN) & (voltage < MAX)]

    # Find a normal distribution for the data using LMFIT
    model = lm.models.GaussianModel()
    params_normal = model.guess(y_data, x=x_data)
    result_normal = model.fit(y_data, params_normal, x=x_data)
    print(result_normal.fit_report())

    # Find a normal distribution + linear for the data using LMFIT
    def special_gaussian(x, A, mu, sigma, m, c):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + m * x + c
    model = lm.Model(special_gaussian)
    params_special = model.make_params(A=100, mu=6.5, sigma=0.1, m=0, c=0)
    result_special = model.fit(y_data, params_special, x=x_data)
    print(result_special.fit_report())
    
    # Plot the fits
    plt.plot(PHOT_X, result_normal.eval(x=PHOT_X), label='Gaussian Fit', color='red')
    plt.plot(PHOT_X, result_special.eval(x=PHOT_X), label='Gaussian + Linear Fit', color='blue')

    # Show plot
    plt.legend()
    plt.show()


# Manual caesium energy data
if ENERGY_MANUAL_CAESIUM:

    # Load manual caesium data
    caesium_data = np.loadtxt('gamma_spectroscopy/manual_caesium.tsv', delimiter='\t', skiprows=1)
    voltage, counts = caesium_data.T
    energy = (0.662/6.826)*voltage

    # Plot manual caesium data
    plt.plot(energy, counts, label='Measured Data', marker='o', linestyle='None', color='black')
    plt.xlabel(r'Energy ($MeV$)')
    plt.ylabel(r'Counts')
    plt.title(r'Manual $^{137}$Cs : Counts vs Energy')
    plt.show()


# Manual caesium data characteristics
if MORE_MANUAL_CAESIUM:

    # Load manual caesium data
    caesium_data = np.loadtxt('gamma_spectroscopy/manual_caesium.tsv', delimiter='\t', skiprows=1)
    voltage, counts = caesium_data.T
    energy = (0.662/6.826)*voltage

    # Plot manual caesium data
    plt.plot(energy, counts, label='Measured Data', marker='o', linestyle='None', color='black')
    plt.xlabel(r'Energy ($MeV$)')
    plt.ylabel(r'Counts')
    plt.title(r'Manual $^{137}$Cs : Counts vs Energy')

    # Pre-fit parameters
    PHOT_X = np.linspace(min(energy), max(energy), 1000)

    # Photopeak
    MIN, MAX = 5.5, 8
    x_data = energy[(voltage > MIN) & (voltage < MAX)]
    y_data = counts[(voltage > MIN) & (voltage < MAX)]
    model = lm.models.GaussianModel()
    params_normal = model.guess(y_data, x=x_data)
    result_normal = model.fit(y_data, params_normal, x=x_data)
    print(result_normal.fit_report())
    plt.plot(PHOT_X, result_normal.eval(x=PHOT_X), label='Photopeak Fit', color='red')

    # Compton edge
    MIN, MAX = 3.5, 5.5
    x_data = energy[(voltage > MIN) & (voltage < MAX)]
    y_data = counts[(voltage > MIN) & (voltage < MAX)]
    def sigmoid(x, A, B, C, D):
        return A / (1 + np.exp(-B * (x - C))) + D
    model = lm.Model(sigmoid)
    params_normal = model.make_params(A=1300, B=0.1, C=0.5, D=400)
    result_normal = model.fit(y_data, params_normal, x=x_data)
    print(result_normal.fit_report())
    plt.plot(PHOT_X, result_normal.eval(x=PHOT_X), label='Compton Edge Fit', color='green')

    # Backscatter peak
    MIN, MAX = 1, 3.5
    x_data = energy[(voltage > MIN) & (voltage < MAX)]
    y_data = counts[(voltage > MIN) & (voltage < MAX)]
    def vertical_gaussian(x, A, mu, sigma, c):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + c
    model = lm.Model(vertical_gaussian)
    params_normal = model.make_params(A=1599, mu=0.19, sigma=0.1, c=2000)
    result_normal = model.fit(y_data, params_normal, x=x_data)
    print(result_normal.fit_report())
    plt.plot(PHOT_X, result_normal.eval(x=PHOT_X), label='Backscatter Peak Fit', color='blue')






    plt.show()

