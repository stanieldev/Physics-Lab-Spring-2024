# Import the necessary packages and modules
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from math import inf as Inf
from lmfit import Model
from scipy.special import wofz



# Create a function that takes in a wavelength and returns the color as RGB
def wavelength_to_rgb(wavelength, gamma=0.8):
    
    # Check if wavelength is iterable
    if hasattr(wavelength, "__iter__"):
        return np.array([wavelength_to_rgb(x) for x in wavelength])

    # Convert wavelength to RGB
    wavelength = float(wavelength)
    if wavelength <= 380:
        attenuation = 0.3
        R = ((attenuation) ** gamma) * np.exp(-5 * ((wavelength - 380) / 50) ** 2)
        G = 0.0
        B = ((attenuation) ** gamma) * np.exp(-5 * ((wavelength - 380) / 50) ** 2)
    elif wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        attenuation = 0.3
        R = ((attenuation) ** gamma) * np.exp(-5 * ((wavelength - 750) / 50) ** 2)
        G = 0.0
        B = 0.0

    # Return the RGB values
    return (R, G, B)

# Def autosave figure function
def save_figure(directory: str) -> None:

    # Save the figure to the quantum_dots directory
    file_name = os.path.splitext(os.path.basename(directory))[0]
    new_file_name = f"./quantum_dots/{file_name}.png"
    if os.path.isfile(new_file_name):
        i = 1
        while True:
            if os.path.isfile(f"./quantum_dots/{file_name}_{i}.png"):
                i += 1
            else:
                plt.savefig(f"./quantum_dots/{file_name}_{i}.png")
                break
    else:
        plt.savefig(new_file_name)



# Define the gaussian function
def gauss(x, A, mu, s):
    return A * np.exp(-(x-mu)**2 / (2*s**2))

# Define the voigt function
def skew_voigt(x, A, mu, s, g, skew):
    return A * np.real(wofz(((x-mu) + 1j*skew)/s/np.sqrt(2))) / s / np.sqrt(2*np.pi) + g * np.exp(-(x-mu)**2 / (2*s**2))

# Lmfit Gaussian fit
def fit_gauss(x, y, dy):

    # Random guess for the parameters
    A = max(y)
    mu = x[np.argmax(y)]
    s = 1

    # Fit the data
    gModel = Model(gauss)
    gModel.set_param_hint('A', value=A, min=0, max=Inf)
    gModel.set_param_hint('mu', value=mu, min=0, max=Inf)
    gModel.set_param_hint('s', value=s, min=0, max=Inf)
    params = gModel.make_params()
    result = gModel.fit(y, x=x, A=A, mu=mu, s=s, weights = 1.0/dy)
    dely = result.eval_uncertainty(sigma=1)
    # plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")

    # Print the results
    print('Lmfit Model fitting results:')
    print(result.fit_report())

    # Return the result
    return result

# Lmfit Voigt fit
def fit_voigt(x, y, dy):

    # Random guess for the parameters
    A = max(y)
    mu = x[np.argmax(y)]
    s = 1
    g = 1
    skew = 0

    # Fit the data
    vModel = Model(skew_voigt)
    vModel.set_param_hint('A', value=A, min=0, max=Inf)
    vModel.set_param_hint('mu', value=mu, min=0, max=Inf)
    vModel.set_param_hint('s', value=s, min=0, max=Inf)
    vModel.set_param_hint('g', value=g, min=0, max=Inf)
    vModel.set_param_hint('skew', value=skew, min=-Inf, max=Inf)
    params = vModel.make_params()
    result = vModel.fit(y, x=x, A=A, mu=mu, s=s, g=g, skew=skew, weights = 1.0/dy)
    dely = result.eval_uncertainty(sigma=1)
    # plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")

    # Print the results
    print('Lmfit Model fitting results:')
    print(result.fit_report())

    # Return the result
    return result



# Define the console program class
class ConsoleProgram:
    def __init__(self, DEBUG=False):
        self.DEBUG = DEBUG
    
    # Define the parse console function
    def parse_console(self):

        # Create the parser
        parser = argparse.ArgumentParser(description="Quantum Dots Console")
        parser.add_argument("file", const=True, nargs="?", type=str, help="Data directory")
        parser.add_argument("-d", "--diameter", const=True, nargs="?", type=float, help="Diameter of the quantum dots in nm")
        parser.add_argument("-l", "--min", const=True, nargs="?", type=float, help="Minimum wavelength in nm")
        parser.add_argument("-u", "--max", const=True, nargs="?", type=float, help="Maximum wavelength in nm")
        parser.add_argument("-m", "--model", nargs="+", type=str, default=[], help="Model to use for fitting")
        parser.add_argument("-ns", "--noshow", const=True, nargs="?", type=bool, default=False, help="Don't show figure option")
        parser.add_argument("-c", "--color", const=True, nargs="?", type=bool, default=False, help="Display color option")
        parser.add_argument("-a", "--autosave", const=True, nargs="?", type=bool, default=False, help="Autosave figure option")
        args = parser.parse_args()

        # If DEBUG, print all args with their value and type
        [print(f"{arg}: {getattr(args, arg)} ({type(getattr(args, arg))})") for arg in vars(args) if self.DEBUG]

        # Check if the file exists
        if not isinstance(args.file, str):
            raise Exception("Directory not specified (required)")
        elif not os.path.exists(args.file):
            raise Exception("Invalid directory (does not exist)")

        # Check if diameter is valid
        if not isinstance(args.diameter, float):
            raise Exception("Diameter not specified (required)")
        elif not float(args.diameter) > 0:
            raise Exception("Invalid diameter (must be greater than 0nm)")

        # If DEBUG, print all args with their value and type
        [print(f"{arg}: {getattr(args, arg)} ({type(getattr(args, arg))})") for arg in vars(args) if self.DEBUG]

        # Sets variables to the class
        self.file = args.file
        self.diameter = args.diameter
        self.min = args.min
        self.max = args.max
        self.model = args.model
        self.show = not args.noshow
        self.color = args.color
        self.autosave = args.autosave

        # Load the data into the program
        self._load_data()

        # Check if minimum wavelength is valid
        if self.min is None:
            self.min = min(self.x_data)
        elif float(args.min) < min(self.x_data):
            print("Warning: Minimum wavelength is less than the minimum wavelength in the data file.")

        # Check if maximum wavelength is valid
        if self.max is None:
            self.max = max(self.x_data)
        elif float(args.max) > max(self.x_data):
            print("Warning: Maximum wavelength is greater than the maximum wavelength in the data file.")
    
    # Define the load data function
    def _load_data(self):

        # Open the file
        with open(self.file, "r") as f:
            data = f.readlines()
        
        # Remove lines that don't start with a number (non-data lines)
        data = [x for x in data if x[0].isdigit()]

        # Get the wavelength and intensity
        wavelength = np.array([float(x.split()[0]) for x in data])
        intensity = np.array([float(x.split()[1]) for x in data])

        # Return the wavelength and intensity
        self.x_data = wavelength
        self.y_data = intensity
        self.y_data_error = np.sqrt(np.abs(5 * intensity)) / 5

    # Define the filter data function
    def _filtered_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_data[np.where((self.x_data > self.min) & (self.x_data < self.max))], \
               self.y_data[np.where((self.x_data > self.min) & (self.x_data < self.max))]

    # Define the plot data function
    def plot_data(self) -> None:

        # Set up the graph
        plt.title(f"Emission Intensity vs Wavelength ({self.diameter}nm Diameter)")
        plt.ylabel("Intensity (counts per 500 ms)")
        plt.xlabel("Wavelength (nm)")

        # If there are no models specified, filter the data and plot it
        if len(self.model) == 0:
            self.x_data, self.y_data = self._filtered_data()
            self.y_data_error = np.sqrt(np.abs(5 * self.y_data)) / 5

        # Set the x and y limits
        plt.xlim(min(self.x_data), max(self.x_data))

        # If color is true, plot the data using the wavelength_to_rgb function
        if self.color:
            [plt.errorbar(w, i, yerr=di, fmt=".", color=c) for w, i, di, c in zip(self.x_data, self.y_data, self.y_data_error, wavelength_to_rgb(self.x_data))]
        else:
            plt.errorbar(self.x_data, self.y_data, yerr=self.y_data_error, fmt=".", label=f"{self.diameter}nm Data")

        # If models are specified, plot the models
        if len(self.model) > 0:

            # Filter the data to the specified range
            x_data, y_data = self._filtered_data()
            y_data_error = np.sqrt(np.abs(5 * y_data)) / 5

            # Plot the regressions if specified
            if "gauss" in self.model:

                # Find the best fit
                result = fit_gauss(x_data, y_data, y_data_error)

                # Plot the gaussian and peak using the parameters
                COLOR = np.random.rand(3,)
                x = np.linspace(min(self.x_data), max(self.x_data), 1000)
                plt.plot(x, gauss(x, result.best_values["A"], result.best_values["mu"], result.best_values["s"]), label=f"Gaussian Distribution", color=COLOR, zorder=3)
                plt.axvline(result.best_values["mu"], linestyle="--", label=f"λ±dλ = {result.best_values['mu']:.1f}±{result.params['mu'].stderr:.1f}nm", color=COLOR)

            if "voigt" in self.model:

                # Find the best fit
                result = fit_voigt(x_data, y_data, y_data_error)
                
                # Plot the gaussian and peak using the parameters
                COLOR = np.random.rand(3,)
                x = np.linspace(min(self.x_data), max(self.x_data), 1000)
                plt.plot(x, skew_voigt(x, result.best_values["A"], result.best_values["mu"], result.best_values["s"], result.best_values["g"], result.best_values["skew"]), label=f"Voigt Distribution", color=COLOR, zorder=3)
                plt.axvline(result.best_values["mu"], linestyle="--", label=f"λ±dλ = {result.best_values['mu']:.1f}±{result.params['mu'].stderr:.1f}nm", color=COLOR)

            # Add a legend
            plt.legend()
        
        # If autosave is true, save the figure
        if self.autosave:
            save_figure(self.file)

        # If show is true, show the plot
        if self.show:
            plt.show(block=True)
            



# Define the main function
def main():
    program = ConsoleProgram(DEBUG=False)
    program.parse_console()
    program.plot_data()

# This is the standard boilerplate that calls the main() function.
if __name__ == "__main__":
    main()
