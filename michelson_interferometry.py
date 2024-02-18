# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm

# Constants
L0 = 64.66e+6     # Initial copper rod length in nm
dL0 = 0.01e+6     # Uncertainty in initial copper rod length in nm
LAMBDA = 627.9    # Wavelength of laser in nm
dLAMBDA = 5.4     # Uncertainty in wavelength of laser in nm
L_FACTOR = np.sqrt((dL0 / L0)**2 + (dLAMBDA / LAMBDA)**2)  # Factor for uncertainty in L0 and LAMBDA
L_GAS = 100e+6     # Length of gas chamber in nm
dL_GAS = 0.1e+6    # Uncertainty in length of gas chamber in nm
print("L_FACTOR:", L_FACTOR)



# Make a class that holds the data
class Data:
    def __init__(self, cause, response, response_error, name="Data"):
        self.x = cause
        self.y = response
        self.dy = response_error
        self.name = name
    
    def plot(self):
        plt.errorbar(self.x, self.y, self.dy, fmt="o", label=self.name, markersize=2)
        
    def inverted_plot(self):
        plt.errorbar(self.y, self.x, self.dy, fmt="o", label=self.name, markersize=2)

    def plot_linear(self):
        model = lm.models.LinearModel()
        params = model.guess(self.y, x=self.x)
        result = model.fit(self.y, params, x=self.x)  # , weights=weight
        print(self.name + " Fit")
        print(result.fit_report())
        plt.plot(self.x, result.best_fit, "r-", label=self.name + " Fit", color=np.random.rand(3,))
    
    def inv_plot_linear(self):
        model = lm.models.LinearModel()
        params = model.guess(self.x, x=self.y)
        result = model.fit(self.x, params, x=self.y)  # , weights=weight
        print(self.name + " Fit")
        print(result.fit_report())
        plt.plot(self.y, result.best_fit, "r-", label=self.name + " Fit", color=np.random.rand(3,))





# Copper Heating Data
SHOW_HOT_COPPER = False
if SHOW_HOT_COPPER:

    # Load heating data from file
    data = np.loadtxt("michelson_interferometry/copper_heating.txt", skiprows=1, delimiter="\t")
    copper_heating_1 = Data(data[:, 0], data[:, 1], [1e-1 for _ in range(len(data[:, 1]))], "Heating Recording 1")
    copper_heating_2 = Data(data[:, 0], data[:, 2], [1e-1 for _ in range(len(data[:, 1]))], "Heating Recording 2")

    # Plot the heating data
    plt.title("Heating Copper : Counts vs. Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Counts")
    copper_heating_1.inverted_plot()
    copper_heating_2.inverted_plot()
    plt.legend()
    plt.show()

# Copper Heating Length Data
SHOW_HOT_COPPER_LENGTH = False
if SHOW_HOT_COPPER_LENGTH and SHOW_HOT_COPPER:

    # Find the length differential for the copper rod
    length_differential_1 = Data(copper_heating_1.y, [(N * LAMBDA / 2 / L0) for N in copper_heating_1.x], [(LAMBDA / 2 / L0) for _ in copper_heating_1.x], name="Length Differential")
    length_differential_2 = Data(copper_heating_2.y, [(N * LAMBDA / 2 / L0) for N in copper_heating_2.x], [(LAMBDA / 2 / L0) for _ in copper_heating_2.x], name="Length Differential")

    # Plot the length differential
    plt.title("Heating Copper : Length Differential vs. Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Length Differential (unitless)")
    length_differential_1.plot()
    length_differential_2.plot()
    length_differential_1.plot_linear()
    length_differential_2.plot_linear()
    plt.legend()
    plt.show()








# Copper Cooling Data
SHOW_COOL_COPPER = False
if SHOW_COOL_COPPER:

    # Load cooling data from file
    data = np.loadtxt("michelson_interferometry/copper_cooling.txt", skiprows=1, delimiter="\t")
    copper_cooling_1 = Data(data[:, 0], data[:, 1], [1e-1 for _ in range(len(data[:, 1]))], "Cooling Recording 1")
    copper_cooling_2 = Data(data[:, 0], data[:, 2], [1e-1 for _ in range(len(data[:, 1]))], "Cooling Recording 2")

    # Plot the cooling data
    plt.title("Counts vs. Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Counts")
    copper_cooling_1.inverted_plot()
    copper_cooling_2.inverted_plot()
    plt.legend()
    plt.show()

# Copper Heating Length Data
SHOW_COOL_COPPER_LENGTH = False
if SHOW_COOL_COPPER_LENGTH and SHOW_COOL_COPPER:

    # Find the length differential for the copper rod
    length_differential_1 = Data(copper_cooling_1.y, [(N * LAMBDA / 2 / L0) for N in copper_cooling_1.x], [(LAMBDA / 2 / L0) for _ in copper_cooling_1.x], name="Length Differential")
    length_differential_2 = Data(copper_cooling_2.y, [(N * LAMBDA / 2 / L0) for N in copper_cooling_2.x], [(LAMBDA / 2 / L0) for _ in copper_cooling_2.x], name="Length Differential")

    # Plot the length differential
    plt.title("Length Differential vs. Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Length Differential (unitless)")
    length_differential_1.plot()
    length_differential_2.plot()
    length_differential_1.plot_linear()
    length_differential_2.plot_linear()
    plt.legend()
    plt.show()



# Gas Pulling Data
SHOW_GAS_PULL = False
if SHOW_GAS_PULL:

    # Load gas pulling data
    data_1 = np.loadtxt("michelson_interferometry/gas_vacuum_1.txt", skiprows=1, delimiter="\t")
    data_2 = np.loadtxt("michelson_interferometry/gas_vacuum_2.txt", skiprows=1, delimiter="\t")
    data_3 = np.loadtxt("michelson_interferometry/gas_vacuum_3.txt", skiprows=1, delimiter="\t")
    data_4 = np.loadtxt("michelson_interferometry/gas_vacuum_4.txt", skiprows=1, delimiter="\t")
    data_5 = np.loadtxt("michelson_interferometry/gas_vacuum_5.txt", skiprows=1, delimiter="\t")
    gas_vacuum_1 = Data(data_1[:, 0], data_1[:, 1]/10+1, [1e-3 for _ in range(len(data_1[:, 1]))], "Gas Vacuum Recording 1")
    gas_vacuum_2 = Data(data_2[:, 0], data_2[:, 1]/10+1, [1e-3 for _ in range(len(data_2[:, 1]))], "Gas Vacuum Recording 2")
    gas_vacuum_3 = Data(data_3[:, 0], data_3[:, 1]/10+1, [1e-3 for _ in range(len(data_3[:, 1]))], "Gas Vacuum Recording 3")
    gas_vacuum_4 = Data(data_4[:, 0], data_4[:, 1]/10+1, [1e-3 for _ in range(len(data_4[:, 1]))], "Gas Vacuum Recording 4")
    gas_vacuum_5 = Data(data_5[:, 0], data_5[:, 1]/10+1, [1e-3 for _ in range(len(data_5[:, 1]))], "Gas Vacuum Recording 5")

    # Plot the gas pulling data
    plt.title("Counts vs. Pressure (atm)")
    plt.xlabel("Pressure (atm)")
    plt.ylabel("Counts")
    gas_vacuum_1.inverted_plot()
    gas_vacuum_2.inverted_plot()
    gas_vacuum_3.inverted_plot()
    gas_vacuum_4.inverted_plot()
    gas_vacuum_5.inverted_plot()
    plt.legend()
    plt.show()

# Gas Pulling Refraction Data
SHOW_GAS_PULL_REFRACTION = False
if SHOW_GAS_PULL_REFRACTION and SHOW_GAS_PULL:

    # Find the refractive index for the gas
    n_1 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_vacuum_1.x]
    n_2 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_vacuum_2.x]
    n_3 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_vacuum_3.x]
    n_4 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_vacuum_4.x]
    n_5 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_vacuum_5.x]
    N_factor = np.sqrt((dLAMBDA / LAMBDA)**2 + (dL_GAS / L_GAS)**2)
    print("refractive index error factor: ", N_factor)
    dn_1 = [0 for n in n_1]  # n * N_factor
    dn_2 = [0 for n in n_2]  # n * N_factor
    dn_3 = [0 for n in n_3]  # n * N_factor
    dn_4 = [0 for n in n_4]  # n * N_factor
    dn_5 = [0 for n in n_5]  # n * N_factor
    refractive_index_1 = Data(gas_vacuum_1.y, n_1, dn_1, name="Gas Refractive Index 1")
    refractive_index_2 = Data(gas_vacuum_2.y, n_2, dn_2, name="Gas Refractive Index 2")
    refractive_index_3 = Data(gas_vacuum_3.y, n_3, dn_3, name="Gas Refractive Index 3")
    refractive_index_4 = Data(gas_vacuum_4.y, n_4, dn_4, name="Gas Refractive Index 4")
    refractive_index_5 = Data(gas_vacuum_5.y, n_5, dn_5, name="Gas Refractive Index 5")

    # Plot the refractive index
    plt.title("Refractive Index vs. Pressure (atm)")
    plt.xlabel("Pressure (atm)")
    plt.ylabel("Refractive Index")
    refractive_index_1.plot()
    refractive_index_2.plot()
    refractive_index_3.plot()
    refractive_index_4.plot()
    refractive_index_5.plot()
    refractive_index_1.plot_linear()
    refractive_index_2.plot_linear()
    refractive_index_3.plot_linear()
    refractive_index_4.plot_linear()
    refractive_index_5.plot_linear()
    plt.legend()
    plt.show()










# Gas Releasing Data
SHOW_GAS_RELEASE = True
if SHOW_GAS_RELEASE:

    # Load gas releasing data
    data_1 = np.loadtxt("michelson_interferometry/gas_release_1.txt", skiprows=1, delimiter="\t")
    data_2 = np.loadtxt("michelson_interferometry/gas_release_2.txt", skiprows=1, delimiter="\t")
    data_3 = np.loadtxt("michelson_interferometry/gas_release_3.txt", skiprows=1, delimiter="\t")
    data_4 = np.loadtxt("michelson_interferometry/gas_release_4.txt", skiprows=1, delimiter="\t")
    data_5 = np.loadtxt("michelson_interferometry/gas_release_5.txt", skiprows=1, delimiter="\t")
    gas_release_1 = Data(data_1[:, 0], data_1[:, 1]/10+1, [1e-3 for _ in range(len(data_1[:, 1]))], "Gas Release Recording 1")
    gas_release_2 = Data(data_2[:, 0], data_2[:, 1]/10+1, [1e-3 for _ in range(len(data_2[:, 1]))], "Gas Release Recording 2")
    gas_release_3 = Data(data_3[:, 0], data_3[:, 1]/10+1, [1e-3 for _ in range(len(data_3[:, 1]))], "Gas Release Recording 3")
    gas_release_4 = Data(data_4[:, 0], data_4[:, 1]/10+1, [1e-3 for _ in range(len(data_4[:, 1]))], "Gas Release Recording 4")
    gas_release_5 = Data(data_5[:, 0], data_5[:, 1]/10+1, [1e-3 for _ in range(len(data_5[:, 1]))], "Gas Release Recording 5")

    # Plot the gas releasing data
    plt.title("Counts vs. Pressure (atm)")
    plt.xlabel("Pressure (atm)")
    plt.ylabel("Counts")
    gas_release_1.inverted_plot()
    gas_release_2.inverted_plot()
    gas_release_3.inverted_plot()
    gas_release_4.inverted_plot()
    gas_release_5.inverted_plot()
    plt.legend()
    plt.show()

# Gas Releasing Refraction Data
SHOW_GAS_RELEASE_REFRACTION = True
if SHOW_GAS_RELEASE_REFRACTION and SHOW_GAS_RELEASE:

    # Find the refractive index for the gas
    n_1 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_release_1.x]
    n_2 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_release_2.x]
    n_3 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_release_3.x]
    n_4 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_release_4.x]
    n_5 = [M*LAMBDA / (2*L_GAS) + 1 for M in gas_release_5.x]
    N_factor = np.sqrt((dLAMBDA / LAMBDA)**2 + (dL_GAS / L_GAS)**2)
    print("refractive index error factor: ", N_factor)
    dn_1 = [0 for n in n_1]  # n * N_factor
    dn_2 = [0 for n in n_2]  # n * N_factor
    dn_3 = [0 for n in n_3]  # n * N_factor
    dn_4 = [0 for n in n_4]  # n * N_factor
    dn_5 = [0 for n in n_5]  # n * N_factor
    refractive_index_1 = Data(gas_release_1.y, n_1, dn_1, name="Gas Refractive Index 1")
    refractive_index_2 = Data(gas_release_2.y, n_2, dn_2, name="Gas Refractive Index 2")
    refractive_index_3 = Data(gas_release_3.y, n_3, dn_3, name="Gas Refractive Index 3")
    refractive_index_4 = Data(gas_release_4.y, n_4, dn_4, name="Gas Refractive Index 4")
    refractive_index_5 = Data(gas_release_5.y, n_5, dn_5, name="Gas Refractive Index 5")

    # Plot the refractive index
    plt.title("Gas Release : Refractive Index vs. Pressure (atm)")
    plt.xlabel("Pressure (atm)")
    plt.ylabel("Refractive Index")
    refractive_index_1.plot()
    refractive_index_2.plot()
    refractive_index_3.plot()
    refractive_index_4.plot()
    refractive_index_5.plot()
    refractive_index_1.plot_linear()
    refractive_index_2.plot_linear()
    refractive_index_3.plot_linear()
    refractive_index_4.plot_linear()
    refractive_index_5.plot_linear()
    plt.legend()
    plt.show()









# # Create a function to plot the data
# def plot_data(x, y, yerr, label):
#     model = lm.models.LinearModel()
#     weight = [1/i for i in yerr]
#     params = model.guess(y, x=x)
#     result = model.fit(y, params, x=x) # , weights=weight
#     print(result.fit_report())
#     plt.plot(x, result.best_fit, "r-", label=label + " Fit", color=np.random.rand(3,))












# # Create lists for the fractional change in length
# fl_t1_hot = [(N * LAMBDA / 2) / L0 for N in c1_hot]
# fl_t2_hot = [(N * LAMBDA / 2) / L0 for N in c2_hot]
# fl_t3_hot = [(N * LAMBDA / 2) / L0 for N in c3_hot]
# fl_t4_hot = [(N * LAMBDA / 2) / L0 for N in c4_hot]
# fl_t5_hot = [(N * LAMBDA / 2) / L0 for N in c5_hot]
# fl_t6_hot = [(N * LAMBDA / 2) / L0 for N in c6_hot]

# fl_t1_cool = [(N * LAMBDA / 2) / L0 for N in c1_cool]
# fl_t2_cool = [(N * LAMBDA / 2) / L0 for N in c2_cool]
# fl_t3_cool = [(N * LAMBDA / 2) / L0 for N in c3_cool]
# fl_t4_cool = [(N * LAMBDA / 2) / L0 for N in c4_cool]
# fl_t5_cool = [(N * LAMBDA / 2) / L0 for N in c5_cool]
# fl_t6_cool = [(N * LAMBDA / 2) / L0 for N in c6_cool]


# # Plot the fractional change in length against temperature
# SHOW = True
# FIT = True
# if SHOW:
#     plt.title("Heating Fractional Change in Length vs. Temperature")
#     plt.xlabel("Temperature (C)")
#     plt.ylabel("Fractional Change in Length")
#     plt.errorbar(t1_hot, fl_t1_hot, [float(L_FACTOR * i) for i in fl_t1_hot], label="Heating Recording 1", fmt="o")
#     plt.errorbar(t2_hot, fl_t2_hot, [float(L_FACTOR * i) for i in fl_t2_hot], label="Heating Recording 2", fmt="o")
#     plt.errorbar(t3_hot, fl_t3_hot, [float(L_FACTOR * i) for i in fl_t3_hot], label="Heating Recording 3", fmt="o")
#     plt.errorbar(t4_hot, fl_t4_hot, [float(L_FACTOR * i) for i in fl_t4_hot], label="Heating Recording 4", fmt="o")
#     plt.errorbar(t5_hot, fl_t5_hot, [float(L_FACTOR * i) for i in fl_t5_hot], label="Heating Recording 5", fmt="o")
#     plt.errorbar(t6_hot, fl_t6_hot, [float(L_FACTOR * i) for i in fl_t6_hot], label="Heating Recording 6", fmt="o")
# if SHOW and FIT:
#     plot_data(t1_hot, fl_t1_hot, [float(L_FACTOR * i) for i in fl_t1_hot], label="Heating Recording 1")
#     plot_data(t2_hot, fl_t2_hot, [float(L_FACTOR * i) for i in fl_t2_hot], label="Heating Recording 2")
#     plot_data(t3_hot, fl_t3_hot, [float(L_FACTOR * i) for i in fl_t3_hot], label="Heating Recording 3")
#     plot_data(t4_hot, fl_t4_hot, [float(L_FACTOR * i) for i in fl_t4_hot], label="Heating Recording 4")
#     plot_data(t5_hot, fl_t5_hot, [float(L_FACTOR * i) for i in fl_t5_hot], label="Heating Recording 5")
#     plot_data(t6_hot, fl_t6_hot, [float(L_FACTOR * i) for i in fl_t6_hot], label="Heating Recording 6")
    
# if SHOW:
#     plt.legend()
#     plt.show()





# # Plot the fractional change in length against temperature
# SHOW = False
# FIT = True
# if SHOW:
#     plt.title("Cooling Fractional Change in Length vs. Temperature")
#     plt.xlabel("Temperature (C)")
#     plt.ylabel("Fractional Change in Length")
#     plt.errorbar(t1_cool, fl_t1_cool, [L_FACTOR * i for i in fl_t1_cool], label="Cooling Recording 1", fmt="o")
#     plt.errorbar(t2_cool, fl_t2_cool, [L_FACTOR * i for i in fl_t2_cool], label="Cooling Recording 2", fmt="o")
#     plt.errorbar(t3_cool, fl_t3_cool, [L_FACTOR * i for i in fl_t3_cool], label="Cooling Recording 3", fmt="o")
#     plt.errorbar(t4_cool, fl_t4_cool, [L_FACTOR * i for i in fl_t4_cool], label="Cooling Recording 4", fmt="o")
#     plt.errorbar(t5_cool, fl_t5_cool, [L_FACTOR * i for i in fl_t5_cool], label="Cooling Recording 5", fmt="o")
#     plt.errorbar(t6_cool, fl_t6_cool, [L_FACTOR * i for i in fl_t6_cool], label="Cooling Recording 6", fmt="o")
# if SHOW and FIT:
#     plot_data(t1_cool, fl_t1_cool, [L_FACTOR * i for i in fl_t1_cool], label="Cooling Recording 1")
#     plot_data(t2_cool, fl_t2_cool, [L_FACTOR * i for i in fl_t2_cool], label="Cooling Recording 2")
#     plot_data(t3_cool, fl_t3_cool, [L_FACTOR * i for i in fl_t3_cool], label="Cooling Recording 3")
#     plot_data(t4_cool, fl_t4_cool, [L_FACTOR * i for i in fl_t4_cool], label="Cooling Recording 4")
#     plot_data(t5_cool, fl_t5_cool, [L_FACTOR * i for i in fl_t5_cool], label="Cooling Recording 5")
#     plot_data(t6_cool, fl_t6_cool, [L_FACTOR * i for i in fl_t6_cool], label="Cooling Recording 6")
# if SHOW:
#     plt.legend()
#     plt.show()







# # Create a length differential list using counts
# length_differential = [(N * LAMBDA / 2) for N in counts]

# # Divide length differential by L0 to get fractional change in length
# fractional_change_in_length = [(L - L0) / L0 for L in length_differential]

# # Plot the fractional change in length against temperature
# plt.title("Fractional change in length against temperature")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Fractional change in length")
# plt.plot(temperatures, fractional_change_in_length)

# # Calculate a linear fit for the data
# model = lm.models.LinearModel()
# params = model.guess(fractional_change_in_length, x=temperatures)
# result = model.fit(fractional_change_in_length, params, x=temperatures)
# print("Note: The slope of the linear fit is the coefficient of thermal expansion.")
# print(result.fit_report())

# # Plot the linear fit
# plt.plot(temperatures, result.best_fit, "r-")
# plt.show()