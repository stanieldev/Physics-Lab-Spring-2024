# Imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.optimize import curve_fit
import os
from lmfit import Model
from math import inf as Inf
import scipy as sp
USE_LMFIT = True


# Define the gaussian function
def gauss(x, A, mu, s):
    return A * np.exp(-(x-mu)**2 / (2*s**2))

# Define the voigt function
def voigt(x, A, mu, s, g):
    return A * np.real(wofz(((x-mu) + 1j*g)/s/np.sqrt(2))) / s / np.sqrt(2*np.pi)

# Lmfit Gaussian fit
def fit_gauss(x, y):

    # Pick a random color
    rand_color = np.random.rand(3,)

    # Generator error
    dy = np.sqrt(np.abs(5 * y)) / 5

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
    plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")

    # Print the results
    print('Lmfit Model fitting results:')
    print(result.fit_report())
    print('    chi-square p-value = ', 1.000 - sp.stats.chi2.cdf(result.chisqr, result.nfree))

    # Plot the fit
    plt.plot(x, result.best_fit, label=f"Gaussian Distribution", color=rand_color)

    # Plot a vertical line at the peak
    plt.axvline(result.best_values["mu"], linestyle="--", label=f"λ±dλ = {result.best_values['mu']:.1f}±{result.params['mu'].stderr:.1f}nm", color=rand_color)

# Lmfit Voigt fit
def fit_voigt(x, y):
    
    # Pick a random color
    rand_color = np.random.rand(3,)

    # Generator error
    dy = np.sqrt(np.abs(5 * y)) / 5

    # Random guess for the parameters
    A = max(y)
    mu = x[np.argmax(y)]
    s = 1
    g = 1

    # Fit the data
    vModel = Model(voigt)
    vModel.set_param_hint('A', value=A, min=0, max=Inf)
    vModel.set_param_hint('mu', value=mu, min=0, max=Inf)
    vModel.set_param_hint('s', value=s, min=0, max=Inf)
    vModel.set_param_hint('g', value=g, min=0, max=Inf)
    params = vModel.make_params()
    result = vModel.fit(y, x=x, A=A, mu=mu, s=s, g=g, weights = 1.0/dy)
    dely = result.eval_uncertainty(sigma=1)
    plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")

    # Print the results
    print('Lmfit Model fitting results:')
    print(result.fit_report())
    print('    chi-square p-value = ', 1.000 - sp.stats.chi2.cdf(result.chisqr, result.nfree))

    # Plot the fit
    plt.plot(x, result.best_fit, label=f"Voigt Distribution", color=rand_color)

    # Plot a vertical line at the peak
    plt.axvline(result.best_values["mu"], linestyle="--", label=f"λ±dλ = {result.best_values['mu']:.1f}±{result.params['mu'].stderr:.1f}nm", color=rand_color)

# Define a fit function
def fit_data_scipy(x, y, model) -> None:

    # Pick a random color
    rand_color = np.random.rand(3,)

    # Find the fit
    if model == gauss:
        popt, pcov = curve_fit(model, x, y, maxfev=10000, p0=[max(y), x[np.argmax(y)], 1])
    elif model == voigt:
        popt, pcov = curve_fit(model, x, y, maxfev=10000, p0=[max(y), x[np.argmax(y)], 1, 1])

    # Determine the chi-squared value
    chi_squared = np.sum((y - model(x, *popt))**2 / model(x, *popt))
    red_chi_squared = chi_squared / (len(x) - len(popt))
    p_value = 1.000 - sp.stats.chi2.cdf(chi_squared, len(x) - len(popt))
    
    # Determine the r^2 value
    r_squared = 1 - (np.sum((y - model(x, *popt))**2) / np.sum((y - np.mean(y))**2))

    # Print the results
    print(f"{model.__name__[0].upper()}{model.__name__[1:]} Distribution")
    print(f"Chi-Squared: {chi_squared:.3f}")
    print(f"Reduced Chi-Squared: {red_chi_squared:.3f}")
    print(f"P-Value: {p_value:.3f}")
    print(f"R-Squared: {r_squared:.3f}")

    # Plot the fit
    plt.plot(x, model(x, *popt), label=f"{model.__name__[0].upper()}{model.__name__[1:]} Distribution", color=rand_color)

    # Plot a vertical line at the peak
    plt.axvline(popt[1], linestyle="--", label=f"λ±dλ = {popt[1]:.1f}±{np.sqrt(pcov[1,1]):.1f}nm", color=rand_color)

# Define the read function
def read_data(file: str) -> tuple[np.ndarray, np.ndarray]:

    # Open the file
    try:
        with open(file, "r") as f:
            data = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError("File not found")
        exit(1)
    
    # Remove lines that don't start with a number
    data = [x for x in data if x[0].isdigit()]

    # Get the wavelength and intensity
    wavelength = np.array([float(x.split()[0]) for x in data])
    intensity = np.array([float(x.split()[1]) for x in data])

    # Return the wavelength and intensity
    return wavelength, intensity

# Define the filter function
def filter(x, y, min_lambda: float, max_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    if min_lambda is None: min_lambda = min(x)
    if max_lambda is None: max_lambda = max(x)
    return x[np.where((x > min_lambda) & (x < max_lambda))], \
           y[np.where((x > min_lambda) & (x < max_lambda))]

# Define save figure function
def save_figure(directory: str) -> None:

    # Ask to save the figure
    save = input("Save figure? (y/n): ")
    if save != "y": return

    # Save the figure to the quantum_dots directory
    file_name = os.path.splitext(os.path.basename(directory))[0]
    new_file_name = f"./quantum_dots/{file_name}.png"
    if os.path.isfile(new_file_name):
        overwrite = input("File already exists. Overwrite? (y/n): ")
        if overwrite == "y":
            plt.savefig(new_file_name)
        else:
            i = 1
            while True:
                if os.path.isfile(f"./quantum_dots/{file_name}_{i}.png"):
                    i += 1
                else:
                    plt.savefig(f"./quantum_dots/{file_name}_{i}.png")
                    break
    else:
        plt.savefig(new_file_name)


# Define the main function
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fit quantum dots data")
    parser.add_argument("file", type=str, nargs="?", help="Data directory")
    parser.add_argument("-d", "--diameter", type=int, nargs="+", help="Diameter of quantum dots")
    parser.add_argument("-min", "--min_lambda", type=float, nargs="+", help="Minimum wavelength")
    parser.add_argument("-max", "--max_lambda", type=float, nargs="+", help="Maximum wavelength")
    parser.add_argument("-m","--model", type=str, nargs="+", default=["gauss"], help="Model to use for fitting")
    args = parser.parse_args()

    # Check if diameter is empty
    if args.diameter is None: raise ValueError("Diameter cannot be empty")
    
    # Load the data
    wavelength, intensity = read_data(args.file)
    wavelength, intensity = filter(wavelength, intensity, min_lambda=args.min_lambda, max_lambda=args.max_lambda)
    intensity_error = np.sqrt(np.abs(5 * intensity)) / 5

    # Check if any models are specified
    if "gauss" in args.model:
        if USE_LMFIT:
            fit_gauss(wavelength, intensity)
        else:
            fit_data_scipy(wavelength, intensity, gauss)
    if "voigt" in args.model:
        if USE_LMFIT:
            fit_voigt(wavelength, intensity)
        else:
            fit_data_scipy(wavelength, intensity, voigt)
    if args.model is not None:
        plt.legend()

    # Plot the data
    plt.errorbar(wavelength, intensity, yerr=intensity_error, fmt=".", label=f"{args.file}")
    plt.title(f"Emission Intensity vs Wavelength ({args.diameter[0]}nm Diameter)")
    plt.ylabel("Intensity (counts per 500 ms)")
    plt.xlabel("Wavelength (nm)")
    save_figure(args.file)
    plt.show(block=True)

    
# Run the main function
if __name__ == "__main__":
    main()