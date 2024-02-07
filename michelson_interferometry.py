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
print("L_FACTOR:", L_FACTOR)



# Create a function to plot the data
def plot_data(x, y, yerr, label):
    model = lm.models.LinearModel()
    weight = [1/i for i in yerr]
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x) # , weights=weight
    print(result.fit_report())
    plt.plot(x, result.best_fit, "r-", label=label + " Fit", color=np.random.rand(3,))

# Create a function to load the data from a file
def load_data(filename):

    # Load data from file
    with open(filename, "r") as file:
        data = file.readlines()[1:]
        data = [line.replace(" ", "") for line in data]

    # Split each line into a list of strings using comma as the delimiter
    data = [line.strip().split(",") for line in data]

    # Transpose the list of lists
    data = list(map(list, zip(*data)))

    # Convert each list of strings into a list of floats
    data = [[x for x in line if x != ""] for line in data]
    data = [[float(x) for x in line] for line in data[1:]]

    # Return a list of lists
    return [[list(range(1, len(t)+1)), t] for t in data]



# Load heating data from file
data = load_data("michelson_interferometry/copper_heating.txt")
c1_hot, t1_hot = data[0]
c2_hot, t2_hot = data[1]
c3_hot, t3_hot = data[2]
c4_hot, t4_hot = data[3]
c5_hot, t5_hot = data[4]
c6_hot, t6_hot = data[5]

# Load cooling data from file
data = load_data("michelson_interferometry/copper_cooling.txt")
c1_cool, t1_cool = data[0]
c2_cool, t2_cool = data[1]
c3_cool, t3_cool = data[2]
c4_cool, t4_cool = data[3]
c5_cool, t5_cool = data[4]
c6_cool, t6_cool = data[5]



# Plot the hot raw data
SHOW = False
if SHOW:
    plt.title("Temperature vs. Counts")
    plt.xlabel("Counts")
    plt.ylabel("Temperature (C)")
    plt.errorbar(c1_hot, t1_hot, [0.1 for _ in range(len(t1_hot))], label="Heating Recording 1", fmt="o")
    plt.errorbar(c2_hot, t2_hot, [0.1 for _ in range(len(t2_hot))], label="Heating Recording 2", fmt="o")
    plt.errorbar(c3_hot, t3_hot, [0.1 for _ in range(len(t3_hot))], label="Heating Recording 3", fmt="o")
    plt.errorbar(c4_hot, t4_hot, [0.1 for _ in range(len(t4_hot))], label="Heating Recording 4", fmt="o")
    plt.errorbar(c5_hot, t5_hot, [0.1 for _ in range(len(t5_hot))], label="Heating Recording 5", fmt="o")
    plt.errorbar(c6_hot, t6_hot, [0.1 for _ in range(len(t6_hot))], label="Heating Recording 6", fmt="o")
    plt.legend()
    plt.show()

# Plot the cool raw data
SHOW = False
if SHOW:
    plt.title("Temperature vs. Counts")
    plt.xlabel("Counts")
    plt.ylabel("Temperature (C)")
    plt.errorbar(c1_cool, t1_cool, [0.1 for _ in range(len(t1_cool))], label="Cooling Recording 1", fmt="o")
    plt.errorbar(c2_cool, t2_cool, [0.1 for _ in range(len(t2_cool))], label="Cooling Recording 2", fmt="o")
    plt.errorbar(c3_cool, t3_cool, [0.1 for _ in range(len(t3_cool))], label="Cooling Recording 3", fmt="o")
    plt.errorbar(c4_cool, t4_cool, [0.1 for _ in range(len(t4_cool))], label="Cooling Recording 4", fmt="o")
    plt.errorbar(c5_cool, t5_cool, [0.1 for _ in range(len(t5_cool))], label="Cooling Recording 5", fmt="o")
    plt.errorbar(c6_cool, t6_cool, [0.1 for _ in range(len(t6_cool))], label="Cooling Recording 6", fmt="o")
    plt.legend()
    plt.show()


# Create lists for the fractional change in length
fl_t1_hot = [(N * LAMBDA / 2) / L0 for N in c1_hot]
fl_t2_hot = [(N * LAMBDA / 2) / L0 for N in c2_hot]
fl_t3_hot = [(N * LAMBDA / 2) / L0 for N in c3_hot]
fl_t4_hot = [(N * LAMBDA / 2) / L0 for N in c4_hot]
fl_t5_hot = [(N * LAMBDA / 2) / L0 for N in c5_hot]
fl_t6_hot = [(N * LAMBDA / 2) / L0 for N in c6_hot]

fl_t1_cool = [(N * LAMBDA / 2) / L0 for N in c1_cool]
fl_t2_cool = [(N * LAMBDA / 2) / L0 for N in c2_cool]
fl_t3_cool = [(N * LAMBDA / 2) / L0 for N in c3_cool]
fl_t4_cool = [(N * LAMBDA / 2) / L0 for N in c4_cool]
fl_t5_cool = [(N * LAMBDA / 2) / L0 for N in c5_cool]
fl_t6_cool = [(N * LAMBDA / 2) / L0 for N in c6_cool]


# Plot the fractional change in length against temperature
SHOW = True
FIT = True
if SHOW:
    plt.title("Heating Fractional Change in Length vs. Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Fractional Change in Length")
    plt.errorbar(t1_hot, fl_t1_hot, [float(L_FACTOR * i) for i in fl_t1_hot], label="Heating Recording 1", fmt="o")
    plt.errorbar(t2_hot, fl_t2_hot, [float(L_FACTOR * i) for i in fl_t2_hot], label="Heating Recording 2", fmt="o")
    plt.errorbar(t3_hot, fl_t3_hot, [float(L_FACTOR * i) for i in fl_t3_hot], label="Heating Recording 3", fmt="o")
    plt.errorbar(t4_hot, fl_t4_hot, [float(L_FACTOR * i) for i in fl_t4_hot], label="Heating Recording 4", fmt="o")
    plt.errorbar(t5_hot, fl_t5_hot, [float(L_FACTOR * i) for i in fl_t5_hot], label="Heating Recording 5", fmt="o")
    plt.errorbar(t6_hot, fl_t6_hot, [float(L_FACTOR * i) for i in fl_t6_hot], label="Heating Recording 6", fmt="o")
if SHOW and FIT:
    plot_data(t1_hot, fl_t1_hot, [float(L_FACTOR * i) for i in fl_t1_hot], label="Heating Recording 1")
    plot_data(t2_hot, fl_t2_hot, [float(L_FACTOR * i) for i in fl_t2_hot], label="Heating Recording 2")
    plot_data(t3_hot, fl_t3_hot, [float(L_FACTOR * i) for i in fl_t3_hot], label="Heating Recording 3")
    plot_data(t4_hot, fl_t4_hot, [float(L_FACTOR * i) for i in fl_t4_hot], label="Heating Recording 4")
    plot_data(t5_hot, fl_t5_hot, [float(L_FACTOR * i) for i in fl_t5_hot], label="Heating Recording 5")
    plot_data(t6_hot, fl_t6_hot, [float(L_FACTOR * i) for i in fl_t6_hot], label="Heating Recording 6")
    
if SHOW:
    plt.legend()
    plt.show()





# Plot the fractional change in length against temperature
SHOW = False
FIT = True
if SHOW:
    plt.title("Cooling Fractional Change in Length vs. Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Fractional Change in Length")
    plt.errorbar(t1_cool, fl_t1_cool, [L_FACTOR * i for i in fl_t1_cool], label="Cooling Recording 1", fmt="o")
    plt.errorbar(t2_cool, fl_t2_cool, [L_FACTOR * i for i in fl_t2_cool], label="Cooling Recording 2", fmt="o")
    plt.errorbar(t3_cool, fl_t3_cool, [L_FACTOR * i for i in fl_t3_cool], label="Cooling Recording 3", fmt="o")
    plt.errorbar(t4_cool, fl_t4_cool, [L_FACTOR * i for i in fl_t4_cool], label="Cooling Recording 4", fmt="o")
    plt.errorbar(t5_cool, fl_t5_cool, [L_FACTOR * i for i in fl_t5_cool], label="Cooling Recording 5", fmt="o")
    plt.errorbar(t6_cool, fl_t6_cool, [L_FACTOR * i for i in fl_t6_cool], label="Cooling Recording 6", fmt="o")
if SHOW and FIT:
    plot_data(t1_cool, fl_t1_cool, [L_FACTOR * i for i in fl_t1_cool], label="Cooling Recording 1")
    plot_data(t2_cool, fl_t2_cool, [L_FACTOR * i for i in fl_t2_cool], label="Cooling Recording 2")
    plot_data(t3_cool, fl_t3_cool, [L_FACTOR * i for i in fl_t3_cool], label="Cooling Recording 3")
    plot_data(t4_cool, fl_t4_cool, [L_FACTOR * i for i in fl_t4_cool], label="Cooling Recording 4")
    plot_data(t5_cool, fl_t5_cool, [L_FACTOR * i for i in fl_t5_cool], label="Cooling Recording 5")
    plot_data(t6_cool, fl_t6_cool, [L_FACTOR * i for i in fl_t6_cool], label="Cooling Recording 6")
if SHOW:
    plt.legend()
    plt.show()







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