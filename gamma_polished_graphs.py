# Imports
import numpy as np
import matplotlib.pyplot as plt



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

    def plot_energy(self, log=False):
        if log: plt.yscale('log')
        plt.scatter(self.energy, self.counts, label='Measured Data', color='black', marker='.')
        plt.title(rf'MCA {self.name} : Counts vs Energy')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()



# Define the data
BA_133 = ElementData(r"$^{133}$Ba", "gamma_spectroscopy/barium_133/barium_133.tsv")
BA_137 = ElementData(r"$^{137}$Ba", "gamma_spectroscopy/barium_137/barium_137.tsv", skiprows=26)
CS_137 = ElementData(r"$^{137}$Cs", "gamma_spectroscopy/caesium_137/caesium.tsv")
CO_60 = ElementData(r"$^{60}$Co", "gamma_spectroscopy/cobalt_60/Co60.tsv", skiprows=26)
MN_54 = ElementData(r"$^{54}$Mn", "gamma_spectroscopy/manganese_54/manganese.tsv")
NA_22 = ElementData(r"$^{22}$Na", "gamma_spectroscopy/sodium_22/Na22.tsv", skiprows=26)
SR_90 = ElementData(r"$^{90}$Sr", "gamma_spectroscopy/strontium_90/sr90.tsv")

# Define the background data
BG = ElementData("Background 1", "gamma_spectroscopy/background/background.tsv")
BG_energy = BG.energy
BG_counts = BG.counts



# Plot Caesium 137
if False:
    background_counts = (1482.61/9144.33) * BG.counts
    CS_137.plot_energy(log=True)
    CS_137.counts -= background_counts
    CS_137.counts[CS_137.counts < 1] = 0
    CS_137.plot_energy(log=True)
    
# Plot Barium 133
if False:
    background_counts = (338.03/9144.33) * BG.counts
    BA_133.plot_energy(log=True)
    BA_133.counts -= background_counts
    BA_133.counts[BA_133.counts < 1] = 0
    BA_133.plot_energy(log=True)

# Plot Mangenese 54
if False:
    background_counts = (518.05/9144.33) * BG.counts
    MN_54.plot_energy(log=True)
    MN_54.counts -= background_counts
    MN_54.counts[MN_54.counts < 1] = 0
    MN_54.plot_energy(log=True)



# Adjusted Background to match the energy scale of the other data
K = 21.6/11.1
ADJ_BG = []
for i in range(0, int(1024)):
    bottom = int(i*K)
    top = bottom + 1
    R = i*K - bottom
    try:
        Y = BG.counts[bottom] * R + BG.counts[top] * (1-R)
    except:
        Y = 0
    ADJ_BG.append(Y)



# Plot Sodium 22
if False:
    background_counts = (497.64/9144.33) * BG.counts
    NA_22.plot_energy(log=True)
    NA_22.counts -= ADJ_BG
    NA_22.counts[NA_22.counts < 1] = 0
    NA_22.plot_energy(log=True)

# Plot Cobalt 60
if False:
    background_counts = (586.76/9144.33) * BG.counts
    CO_60.plot_energy(log=True)
    CO_60.counts -= ADJ_BG
    CO_60.counts[CO_60.counts < 1] = 0
    CO_60.plot_energy(log=True)

# Plot Strontium 90
if False:
    background_counts = (5400.83/9144.33) * BG.counts
    SR_90.plot_energy(log=False)
    SR_90.counts -= ADJ_BG
    SR_90.counts[SR_90.counts < 1] = 0
    SR_90.plot_energy(log=False)
