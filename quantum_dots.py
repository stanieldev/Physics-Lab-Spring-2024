# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Constants
m_e = ELECTRON_MASS_KILOGRAMS = 9.1093837e-31
eV = ELECTRON_VOLT_JOULES = 1.60217662e-19
h = PLANCK_CONSTANT_JOULESEC = 6.62607004e-34
c = SPEED_OF_LIGHT_METERSEC = 299792458


# Materials
class CdSe:
    def __init__(self) -> None:
        self.E_g = 1.74 * ELECTRON_VOLT_JOULES
        self.m_e = 0.13 * ELECTRON_MASS_KILOGRAMS
        self.m_h = 0.45 * ELECTRON_MASS_KILOGRAMS

class InP :
    def __init__(self) -> None:
        self.E_g = 1.34 * ELECTRON_VOLT_JOULES
        self.m_e = 0.08 * ELECTRON_MASS_KILOGRAMS
        self.m_h = 0.60 * ELECTRON_MASS_KILOGRAMS


# Wavelength Models
class LambdaModels:
    def find_model(self, name: str):
        if name == "1D Square Well":
            return self.OneDimSquareWell
        elif name == "3D Square Well":
            return self.ThreeDimSquareWell
        elif name == "3D Square Well with Gap":
            return self.ThreeDimSquareWellGapped
        elif name == "Spherical Well with Gap":
            return self.SphericalWellGapped
        elif name == "Spherical Well with Gap and Reduced Mass":
            return self.SphericalWellGappedReduced
        else:
            raise ValueError("Invalid Model Name")

    def OneDimSquareWell(self, quantum_dot_diameter: float, material=None):
        return (8 * m_e * c * (quantum_dot_diameter)**2) / h
    
    def ThreeDimSquareWell(self, quantum_dot_diameter: float, material=None):
        return (8/3 * m_e * c * (quantum_dot_diameter)**2) / h

    def ThreeDimSquareWellGapped(self, quantum_dot_diameter: float, material: CdSe or InP):
        return (h*c)/(material.E_g + (8/3 * m_e * c * (quantum_dot_diameter)**2) / h)
    
    def SphericalWellGapped(self, quantum_dot_diameter: float, material: CdSe or InP):
        return (h*c)/(material.E_g + (h**2 / (2 * m_e * (quantum_dot_diameter)**2)))

    def SphericalWellGappedReduced(self, quantum_dot_diameter: float, material: CdSe or InP):
        inverse_reduced_mass = (1/material.m_e + 1/material.m_h)
        return (h*c)/(material.E_g + (h**2 / (2 * inverse_reduced_mass * (quantum_dot_diameter)**2)))



# Create default plot
def create_default_plot(diameter_str) -> None:
    plt.title(f"Emission Intensity vs Wavelength ({diameter_str} Diameter)")
    plt.ylabel("Intensity (counts per 500 ms)")
    plt.xlabel("Wavelength (nm)")
    plt.show(block=False)

# Filter data
def filter_data(λ: np.ndarray, I: np.ndarray, dI: np.ndarray, λ_min: float, λ_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return λ[np.where((λ >= λ_min) & (λ <= λ_max))], \
           I[np.where((λ >= λ_min) & (λ <= λ_max))], \
          dI[np.where((λ >= λ_min) & (λ <= λ_max))]

# Get index from user, validate, and add to queue
def prompt_user_data() -> list[str]:

    # Print available data options
    AVAILABLE_DATA = ["All", "500nm", "520nm", "560nm", "600nm", "620nm", "640nm"]
    print("Available data: ")
    [print(f"{i}: {AVAILABLE_DATA[i]}") for i in range(1, len(AVAILABLE_DATA))]
    print("0: All data")

    # Get user input
    while True:
        try:
            # Get index
            user_input = input("Enter the number/wavelength of the data you want to use: ")

            # Check if user input is "all"
            if user_input == "0" or user_input.lower() == "all":
                return AVAILABLE_DATA[1:]

            # Check if user input is an index in the list
            elif user_input in [f"{i}" for i in range(1, len(AVAILABLE_DATA))]:
                return [AVAILABLE_DATA[user_input]]
            
            # Check if user input is a wavelength
            elif user_input in [i[:3] for i in AVAILABLE_DATA[1:]]:
                return [f"{user_input}nm"]
            
            # Invalid input
            else:
                raise ValueError("Invalid Index/Wavelength")
                
        except Exception as e:
            print(e)
            continue

# Get index from user, validate, and add to queue
def prompt_user_model() -> list[str]:

    # Print available data options
    AVAILABLE_MODELS = ["All", "1D Square Well", "3D Square Well", "3D Square Well with Gap", "Spherical Well with Gap", "Spherical Well with Gap and Reduced Mass"]
    print("Available models: ")
    [print(f"{i}: {AVAILABLE_MODELS[i]}") for i in range(1, len(AVAILABLE_MODELS))]
    print("0: All models")
    
    # Get user input
    while True:
        try:
            # Get index
            user_input = int(input("Enter the number of the model you want to use: "))

            # If user input is 0, return all models
            if user_input == 0:
                return AVAILABLE_MODELS[1:]
            
            # If model is between 1 and the length of the list, return the model
            elif user_input in range(1, len(AVAILABLE_MODELS)):
                return [AVAILABLE_MODELS[user_input]]

            # Invalid input
            else:
                raise ValueError("Invalid Index/Wavelength")
                
        except Exception as e:
            print(e)
            continue

# Get min-max range for fitting
def prompt_user_min_max_range() -> tuple[float, float]:
    while True:
        try:
            min_wavelength = float(input("Enter the minimum wavelength (nm) for fitting: "))
            max_wavelength = float(input("Enter the maximum wavelength (nm) for fitting: "))
            break
        except TypeError as e:
            print(e)
            continue
    return min_wavelength, max_wavelength












# Main
def main():

    # Load queue based on user input
    data_queue = prompt_user_data()
    for diameter_str in data_queue:

        # Useful variables
        diameter_meters = float(diameter_str[:-2])
        diameter_nanometers = float(diameter_str[:-2]) * 1e9

        # Read data from file
        wavelength, intensity = np.loadtxt(f"quantum_dots/{diameter_str}.txt", skiprows=1, unpack=True)
        intensity_error = np.sqrt(np.abs(5 * intensity)) / 5

        # First plot (whole domain)
        plt.errorbar(wavelength, intensity, yerr=intensity_error, fmt=".", label=f"{diameter_str} Diameter")
        create_default_plot(diameter_str)

        # Ask user for min-max range for fitting and filter
        lambda_min, lambda_max = prompt_user_min_max_range()
        plt.close()

        # Second plot (filtered domain)
        wavelength, intensity, intensity_error = filter_data(wavelength, intensity, intensity_error, lambda_min, lambda_max)
        plt.errorbar(wavelength, intensity, yerr=intensity_error, fmt=".", label=f"{diameter_str} Diameter")
        create_default_plot(diameter_str)
        
        # Ask user for model
        model_queue = prompt_user_model()
        plt.close()
        for model in model_queue:

            # Find model
            function = LambdaModels().find_model(model)
            
            # Testing
            print(f"{diameter_meters=}")
            print(f"{function(diameter_meters, material=CdSe())=}")

            # Expected peak (nm)
            peak = function(diameter_meters, material=CdSe()) * 1e9
            print(f"Peak: {peak} nm")
            plt.axvline(peak, label=f"{model} Peak")
        
        # Third plot (filtered domain with models)
        plt.errorbar(wavelength, intensity, yerr=intensity_error, fmt=".", label=f"{diameter_str} Diameter")
        plt.legend()
        create_default_plot(diameter_str)
        input("Press enter to continue...")

        

    

# Main guard
if __name__ == "__main__":
    main()
