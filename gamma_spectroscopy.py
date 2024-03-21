# Import relevent libraries
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm



# SCA Control panel
PLOT_MANUAL_CAESIUM = False
ENERGY_MANUAL_CAESIUM = False
MORE_MANUAL_CAESIUM = False

# MCA Control panel
PLOT_MCA_Cs137 = False
PLOT_MCA_Mn54 = False
PLOT_MCA_Ba133 = False
PLOT_MCA_NA22 = False
PLOT_MCA_CO60 = False
PLOT_MCA_SR90 = True
GLOBALMIN = 0  # keV

# Half Life Control panel
HALF_LIFE = False




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
        params = model.make_params(amplitude=1000, center=222, sigma=0.1, voffset=3000)
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
    params = model.make_params(amplitude=50, steepness=1, center=200, voffset=3)
    result = model.fit(y_data, params, x=x_data)
    return result





# MCA Caesium data
if PLOT_MCA_Cs137:

    # Load MCA data
    mca_data = np.loadtxt('gamma_spectroscopy/caesium_137/caesium.tsv', delimiter='\t', skiprows=25)
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
        positron_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(positron_result.fit_report())

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

        plt.plot(PHOT_X, positron_result.eval(x=PHOT_X), label='Photopeak Fit', color='red')
        plt.plot(COMP_X, compton_result.eval(x=COMP_X), label='Compton Edge Fit', color='green')
        plt.plot(BACK_X, backscatter_result.eval(x=BACK_X), label='Backscatter Peak Fit', color='blue')
        

    # Show plot
    plt.legend()
    plt.show()

# MCA Manganese data
if PLOT_MCA_Mn54:

    # Load MCA data
    #mca_data = np.loadtxt('gamma_spectroscopy/manganese_54/manganese.tsv', delimiter='\t', skiprows=25)
    mca_data = np.loadtxt('gamma_spectroscopy/manganese_54_2/manganese.tsv', delimiter='\t', skiprows=25)
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
    SHOW_REGRESSIONS = True
    if SHOW_REGRESSIONS:

        # Photopeak
        MIN, MAX = 730, 930
        positron_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(positron_result.fit_report())

        # Compton edge
        MIN, MAX = 500, 730
        compton_result = find_sigmoid(energy_kev, counts, MIN, MAX)
        print(compton_result.fit_report())

        # Backscatter peak
        MIN, MAX = 170, 250
        backscatter_result = find_offset_gaussian(energy_kev, counts, MIN, MAX)
        print(backscatter_result.fit_report())

        # Plot the fits
        PHOT_X = np.linspace(min(energy_kev), max(energy_kev), 1000)
        BACK_MAX = 2*backscatter_result.params['center'].value - GLOBALMIN
        COMP_MAX = 2*compton_result.params['center'].value - BACK_MAX
        BACK_X = np.linspace(0, BACK_MAX, 100)
        COMP_X = np.linspace(BACK_MAX, COMP_MAX, 100)

        plt.plot(PHOT_X, positron_result.eval(x=PHOT_X), label='Photopeak Fit', color='red')
        plt.plot(COMP_X, compton_result.eval(x=COMP_X), label='Compton Edge Fit', color='green')
        plt.plot(BACK_X, backscatter_result.eval(x=BACK_X), label='Backscatter Peak Fit', color='blue')

    # Show plot
    plt.legend()
    plt.show()

# MCA Barium data
if PLOT_MCA_Ba133:
    
        # Load MCA data
        mca_data = np.loadtxt('gamma_spectroscopy/barium_133/barium_133.tsv', delimiter='\t', skiprows=25)
        energy_kev = mca_data[:, 1][mca_data[:, 1] > GLOBALMIN]
        counts = mca_data[:, 2][mca_data[:, 1] > GLOBALMIN]
    
        # Plot MCA data
        plt.plot(energy_kev, counts, label='Measured Data', color='black')
        plt.xlabel(r'Energy ($keV$)')
        plt.ylabel(r'Counts')
        plt.title(r'MCA $^{133}$Ba : Counts vs Energy')
        plt.xlim(GLOBALMIN, max(energy_kev))
        plt.xticks(np.arange(GLOBALMIN, max(energy_kev), 100))
    
        # Regressions
        SHOW_REGRESSIONS = True
        if SHOW_REGRESSIONS:
    
            # Photopeak
            MIN, MAX = 340, 420
            positron_result = find_gaussian(energy_kev, counts, MIN, MAX)
            print(positron_result.fit_report())
    
            # Compton edge
            MIN, MAX = 150, 255
            compton_result = find_sigmoid(energy_kev, counts, MIN, MAX)
            print(compton_result.fit_report())
    
            # Backscatter peak
            MIN, MAX = 100, 140
            backscatter_result = find_offset_gaussian(energy_kev, counts, MIN, MAX)
            print(backscatter_result.fit_report())
    
            # Plot the fits
            PHOT_X = np.linspace(min(energy_kev), max(energy_kev), 1000)
            BACK_MAX = 2*backscatter_result.params['center'].value - GLOBALMIN
            COMP_MAX = 2*compton_result.params['center'].value - BACK_MAX
            BACK_X = np.linspace(0, BACK_MAX, 100)
            COMP_X = np.linspace(BACK_MAX, COMP_MAX, 100)
    
            plt.plot(PHOT_X, positron_result.eval(x=PHOT_X), label='Photopeak Fit', color='red')
            plt.plot(COMP_X, compton_result.eval(x=COMP_X), label='Compton Edge Fit', color='green')
            plt.plot(BACK_X, backscatter_result.eval(x=BACK_X), label='Backscatter Peak Fit', color='blue')

        # Show plot
        plt.legend()
        plt.show()

# MCA Sodium data
if PLOT_MCA_NA22:

    # Load MCA data
    mca_data = np.loadtxt('gamma_spectroscopy/sodium_22/Na22.tsv', delimiter='\t', skiprows=26)
    energy_kev = mca_data[:, 1][mca_data[:, 1] > GLOBALMIN]
    counts = mca_data[:, 2][mca_data[:, 1] > GLOBALMIN]

    # Plot MCA data
    plt.plot(energy_kev, counts, label='Measured Data', color='black')
    plt.xlabel(r'Energy ($keV$)')
    plt.ylabel(r'Counts')
    plt.title(r'MCA $^{22}$Na : Counts vs Energy')
    plt.xlim(GLOBALMIN, max(energy_kev))
    plt.xticks(np.arange(GLOBALMIN, max(energy_kev), 300))

    # Regressions
    SHOW_REGRESSIONS = True
    if SHOW_REGRESSIONS:

        # Positron annihilation peak
        MIN, MAX = 430, 600
        positron_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(positron_result.fit_report())

        # Photopeak
        MIN, MAX = 1200, 1400
        photopeak_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(photopeak_result.fit_report())

        # Coincidence peak
        MIN, MAX = 1700, 1900
        coincidence_result = find_gaussian(energy_kev, counts, MIN, MAX)
        print(coincidence_result.fit_report())

        # Plot the fits
        PHOT_X = np.linspace(min(energy_kev), max(energy_kev), 1000)
        plt.plot(PHOT_X, positron_result.eval(x=PHOT_X), label='Positron Annihilation Peak Fit', color='red')
        plt.plot(PHOT_X, photopeak_result.eval(x=PHOT_X), label='Photopeak Fit', color='green')
        plt.plot(PHOT_X, coincidence_result.eval(x=PHOT_X), label='Coincidence Peak Fit', color='blue')

    # Show plot
    plt.legend()
    plt.show()

# MCA Cobalt data
if PLOT_MCA_CO60:
    
        # Load MCA data
        mca_data = np.loadtxt('gamma_spectroscopy/cobalt_60/Co60.tsv', delimiter='\t', skiprows=26)
        energy_kev = mca_data[:, 1][mca_data[:, 1] > GLOBALMIN]
        counts = mca_data[:, 2][mca_data[:, 1] > GLOBALMIN]
    
        # Plot MCA data
        plt.plot(energy_kev, counts, label='Measured Data', color='black')
        plt.xlabel(r'Energy ($keV$)')
        plt.ylabel(r'Counts')
        plt.title(r'MCA $^{60}$Co : Counts vs Energy')
        plt.xlim(GLOBALMIN, max(energy_kev))
        plt.xticks(np.arange(GLOBALMIN, max(energy_kev), 300))
    
        # Regressions
        SHOW_REGRESSIONS = True
        if SHOW_REGRESSIONS:
    
            # Positron annihilation peak
            MIN, MAX = 170, 550
            positron_result = find_offset_gaussian(energy_kev, counts, MIN, MAX)
            print(positron_result.fit_report())
    
            # Photopeak
            MIN, MAX = 1100, 1260
            photopeak_result = find_gaussian(energy_kev, counts, MIN, MAX)
            print(photopeak_result.fit_report())
    
            # Photopeak 2
            MIN, MAX = 1260, 1470
            photopeak_result_2 = find_gaussian(energy_kev, counts, MIN, MAX)
            print(photopeak_result_2.fit_report())
    
            # Plot the fits
            PHOT_X = np.linspace(min(energy_kev), max(energy_kev), 1000)
            plt.plot(PHOT_X, positron_result.eval(x=PHOT_X), label='Positron Annihilation Peak Fit', color='red')
            plt.plot(PHOT_X, photopeak_result.eval(x=PHOT_X), label='Photopeak 1 Fit', color='green')
            plt.plot(PHOT_X, photopeak_result_2.eval(x=PHOT_X), label='Photopeak 2 Fit', color='blue')
    
        # Show plot
        plt.legend()
        plt.show()

# MCA Strontium data
if PLOT_MCA_SR90:
         
        # Load MCA data
        mca_data = np.loadtxt('gamma_spectroscopy/strontium_90/sr90.tsv', delimiter='\t', skiprows=26)
        energy_kev = mca_data[:, 1][mca_data[:, 1] > GLOBALMIN]
        counts = mca_data[:, 2][mca_data[:, 1] > GLOBALMIN]
    
        # Plot MCA data
        plt.plot(energy_kev, counts, label='Measured Data', color='black')
        plt.xlabel(r'Energy ($keV$)')
        plt.ylabel(r'Counts')
        plt.title(r'MCA $^{90}$Sr : Counts vs Energy')
        plt.xlim(GLOBALMIN, max(energy_kev))
        plt.xticks(np.arange(GLOBALMIN, max(energy_kev), 300))
    
        # Regressions
        SHOW_REGRESSIONS = False
        if SHOW_REGRESSIONS:
            pass

        # Show plot
        plt.legend()
        plt.show()







# Half life of Barium* 137
if HALF_LIFE:
    
    # Load MCA data
    mca_data = np.loadtxt('gamma_spectroscopy/barium_137/barium_137.tsv', delimiter='\t', skiprows=26)
    counts = mca_data[:, 2]
    time = mca_data[:, 0]

    # Plot MCA data
    plt.plot(time, counts, label='Measured Data', color='black')
    plt.xlabel(r'Time ($s$)')
    plt.ylabel(r'Counts')
    plt.title(r'MCA $^{137}$Ba* Decay : Counts vs Time')
    plt.xlim(0, max(time))
    plt.xticks(np.arange(0, max(time), 100))

    # Regressions
    SHOW_REGRESSIONS = True
    if SHOW_REGRESSIONS:
        def exp_decay(t, N0, tau): return N0 * np.exp(-t / tau)
        model = lm.Model(exp_decay)
        params = model.make_params(N0=2000, tau=200)
        result = model.fit(counts, params, t=time)
        print(result.fit_report())
        plt.plot(time, result.eval(t=time), label='Exponential Decay Fit', color='red')

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

