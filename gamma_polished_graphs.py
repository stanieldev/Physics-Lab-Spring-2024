# Imports
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm



def sigmoid(x, amplitude, steepness, center, voffset):
    return amplitude / (1 + np.exp(-steepness * (x - center))) + voffset

def offset_gaussian(x, amplitude, center, sigma, voffset):
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2)) + voffset

def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))





# ElementData class
class ElementData:
    def __init__(self, name, filename, skiprows=25):
        data = self.load_data(filename, skiprows)
        self.channel = data[:,0]
        self.energy = data[:,1]
        self.counts = data[:,2]
        self.name = name
    
    def load_data(self, filename, skiprows):
        return np.loadtxt(filename, delimiter='\t', skiprows=skiprows)

    def plot_channels(self, log=False):
        if log: plt.yscale('log')
        plt.scatter(self.channel, self.counts, label='Measured Data', color='black', marker='.')
        plt.title(rf'MCA {self.name} : Counts vs Channel')
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()

    def plot_energy(self, log=False, color="black"):
        if log: plt.yscale('log')
        plt.scatter(self.energy, self.counts, label='Measured Data', color=color, marker='.')
        plt.title(rf'MCA {self.name} : Counts vs Energy')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

    def plot_adjusted_energy(self, background: list, log=False, color="black"):
        counts = self.counts.copy()
        counts -= background
        counts[counts < 1] = 0
        if log: plt.yscale('log')
        plt.scatter(self.energy, counts, label='Adjusted Measured Data', color=color, marker='.')
        plt.title(rf'MCA {self.name} : Counts vs Energy')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

    def find_compton_edge(self, min_energy, max_energy, p0):
        
        # Filter data
        x_data = self.energy[(self.energy > min_energy) & (self.energy < max_energy)]
        y_data = self.counts[(self.energy > min_energy) & (self.energy < max_energy)]

        # Find the compton edge
        model = lm.Model(sigmoid)
        params = model.make_params(amplitude=p0[0], steepness=p0[1], center=p0[2], voffset=p0[3])
        result = model.fit(y_data, params, x=x_data)
        return result

    def plot_compton_edge(self, min_energy, max_energy, p0, label="Compton Edge", color="green"):
        
        # Find the compton edge
        result = self.find_compton_edge(min_energy, max_energy, p0)
        print(result.fit_report())
        
        # Plot the compton edge
        params = result.best_values
        x_fit = np.linspace(min_energy, max_energy, 1000)
        y_fit = sigmoid(x_fit, params['amplitude'], params['steepness'], params['center'], params['voffset'])
        plt.plot(x_fit, y_fit, label=label, color=color)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

    def find_backscatter_peak(self, min_energy, max_energy, p0):
        
        # Filter data
        x_data = self.energy[(self.energy > min_energy) & (self.energy < max_energy)]
        y_data = self.counts[(self.energy > min_energy) & (self.energy < max_energy)]

        # Find the backscatter peak
        model = lm.Model(offset_gaussian)
        params = model.make_params(amplitude=p0[0], center=p0[1], sigma=p0[2], voffset=p0[3])
        result = model.fit(y_data, params, x=x_data)
        return result

    def plot_backscatter_peak(self, min_energy, max_energy, p0, label="Backscatter Peak", color="red"):

        # Find the backscatter peak
        result = self.find_backscatter_peak(min_energy, max_energy, p0)
        print(result.fit_report())
        
        # Plot the backscatter peak
        params = result.best_values
        x_fit = np.linspace(min_energy, max_energy, 1000)
        y_fit = offset_gaussian(x_fit, params['amplitude'], params['center'], params['sigma'], params['voffset'])
        plt.plot(x_fit, y_fit, label=label, color=color)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

    def find_photopeak(self, min_energy, max_energy, p0):
        
        # Filter data
        x_data = self.energy[(self.energy > min_energy) & (self.energy < max_energy)]
        y_data = self.counts[(self.energy > min_energy) & (self.energy < max_energy)]

        # Find the photopeak
        
        model = lm.Model(gaussian)
        params = model.make_params(amplitude=p0[0], center=p0[1], sigma=p0[2])
        result = model.fit(y_data, params, x=x_data)
        return result
    
    def plot_photopeak(self, min_energy, max_energy, p0, label="Photopeak", color="blue"):
        
        # Find the photopeak
        result = self.find_photopeak(min_energy, max_energy, p0)
        print(result.fit_report())
        
        # Plot the photopeak
        params = result.best_values
        x_fit = np.linspace(min_energy, max_energy, 1000)
        y_fit = gaussian(x_fit, params['amplitude'], params['center'], params['sigma'])
        plt.plot(x_fit, y_fit, label=label, color=color)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

    


# Low Energy Background
BG_LOW = ElementData("Low Energy Background", "gamma_spectroscopy/background_low/background.tsv")
CS_137 = ElementData(r"$^{137}$Cs", "gamma_spectroscopy/caesium_137/caesium.tsv")
BA_133 = ElementData(r"$^{133}$Ba", "gamma_spectroscopy/barium_133/barium_133.tsv")
MN_54 = ElementData(r"$^{54}$Mn", "gamma_spectroscopy/manganese_54/manganese.tsv")

# High Energy Background
BG_HIGH = ElementData("High Energy Background", "gamma_spectroscopy/background_high/background.tsv")
NA_22 = ElementData(r"$^{22}$Na", "gamma_spectroscopy/sodium_22/Na22.tsv", skiprows=26)
CO_60 = ElementData(r"$^{60}$Co", "gamma_spectroscopy/cobalt_60/Co60.tsv", skiprows=26)
SR_90 = ElementData(r"$^{90}$Sr", "gamma_spectroscopy/strontium_90/sr90.tsv")





# Define the data
BA_137 = ElementData(r"$^{137}$Ba", "gamma_spectroscopy/barium_137/barium_137.tsv", skiprows=26)





# Plot Caesium 137
if False:
    CS_137.plot_adjusted_energy((1482.61/9144.33) * BG_LOW.counts, log=True)
    CS_137.plot_backscatter_peak(180, 240, [10000, 200, 10, 5000])
    CS_137.plot_compton_edge(400, 610, [4000, 0.1, 511, 1350])
    CS_137.plot_photopeak(630, 750, [10000, 680, 10])
    plt.show()
    
# Plot Barium 133
if False:
    BA_133.plot_adjusted_energy((338.03/9144.33) * BG_LOW.counts, log=True)
    BA_133.plot_backscatter_peak(105, 141, [6000, 125, 10, 3600])
    BA_133.plot_compton_edge(160, 260, [3000, 0.1, 213, 1200])
    BA_133.plot_photopeak(340, 380, [9610, 365, 10])
    plt.show()

# Plot Mangenese 54
if False:
    MN_54.plot_adjusted_energy((518.05/9144.33) * BG_LOW.counts, log=True)
    MN_54.plot_backscatter_peak(190, 420, [150, 220, 10, 71])
    MN_54.plot_compton_edge(590, 780, [100, 0.1, 690, 20])
    MN_54.plot_photopeak(820, 960, [350, 881, 10])
    plt.show()



# Plot Sodium 22
if False:
    NA_22.plot_adjusted_energy((497.64/9144.33) * BG_HIGH.counts, log=True)
    NA_22.plot_backscatter_peak(135, 276, [10000, 175, 10, 2400], label="Backscatter Peak 1")
    NA_22.plot_compton_edge(280, 430, [10000, 0.1, 330, 2400], label="Compton Edge 1")
    NA_22.plot_backscatter_peak(430, 550, [30000, 510, 10, 2400], label="Photopeak 1", color="blue")

    NA_22.plot_backscatter_peak(635, 800, [600, 702, 10, 1200], label="Backscatter Peak 2")
    NA_22.plot_compton_edge(980, 1200, [1200, 0.1, 1090, 300], label="Compton Edge 2")
    NA_22.plot_backscatter_peak(1203, 1390, [1000, 1287, 10, 330], label="Photopeak 2", color="blue")


    NA_22.plot_backscatter_peak(1720, 1930, [200, 1820, 10, 13], label="Photopeak 3", color="blue")
    plt.show()

# Plot Cobalt 60
if False:
    CO_60.plot_adjusted_energy((586.76/9144.33) * BG_HIGH.counts, log=True)

    CO_60.plot_backscatter_peak(180, 300, [2500, 205, 10, 1400], label="Backscatter Peak 1")
    CO_60.plot_compton_edge(900, 1030, [600, 0.1, 950, 1100], label="Compton Edge 1")
    CO_60.plot_backscatter_peak(1120, 1250, [2500, 1174, 10, 900], label="Photopeak 1", color="blue")

    CO_60.plot_backscatter_peak(1270, 1420, [2500, 1332, 10, 100], label="Backscatter Peak 2")
    CO_60.plot_compton_edge(2100, 2380, [20, 2285, 10, 4], label="Compton Edge 2")
    CO_60.plot_backscatter_peak(2370, 2610, [44, 2506, 10, 3], label="Photopeak 2", color="blue")

    plt.show()

# Plot Strontium 90
if False:
    SR_90.plot_adjusted_energy((5400.83/9144.33) * BG_HIGH.counts, log=False)
    plt.show()




