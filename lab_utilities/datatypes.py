from enum import Enum
import re
from decimal import Decimal
from typing import Union
RAW_UNITS = False




# Create new exception called UnitsError
class UnitsError(Exception): pass

# List of SI prefix multipliers
class _Prefixes:
    y = 1e-24; Y = 1e+24
    z = 1e-21; Z = 1e+21
    a = 1e-18; E = 1e+18
    f = 1e-15; P = 1e+15
    p = 1e-12; T = 1e+12
    n = 1e-09; G = 1e+09
    u = 1e-06; M = 1e+06
    m = 1e-03; k = 1e+03
    c = 1e-02; h = 1e+02
    d = 1e-01; da= 1e+01

# SI Unit storage class
class _Units:
    __slots__ = ["symbol", "kg", "m", "s", "A", "K", "mol", "cd"]

    # Define the _Units class
    def __init__(self, symbol: str = None, kg=0, m=0, s=0, A=0, K=0, mol=0, cd=0) -> None:
        self.symbol: str = symbol
        self.kg:  int = kg
        self.m:   int = m
        self.s:   int = s
        self.A:   int = A
        self.K:   int = K
        self.mol: int = mol
        self.cd:  int = cd
    
    # Define the __repr__ method for the _Units class
    def __repr__(self) -> str:

        # If not RAW_UNITS, return the symbol
        if not RAW_UNITS: return self.symbol

        # Otherwise, return the raw name
        name = ""
        for unit in self.__slots__[1:]:
            if getattr(self, unit) == 1: name += f"{unit} "
            elif getattr(self, unit) != 0: name += f"{unit}^{getattr(self, unit)} "
        return name.strip()

    # Unit multiplication
    def __mul__(self, other: "_Units") -> "_Units":

        # Check if the other object is a _Units object
        if not isinstance(other, _Units): raise ValueError()

        # Create a new _Units object
        new_units = _Units(f"{self.symbol} {other.symbol}")
        for unit in self.__slots__[1:]:
            setattr(new_units, unit, getattr(self, unit) + getattr(other, unit))

        # Return the new _Units object
        return new_units
    
    # Unit division
    def __truediv__(self, other: "_Units") -> "_Units":

        # Check if the other object is a _Units object
        if not isinstance(other, _Units): raise ValueError()

        # Create a new _Units object
        new_units = _Units(f"{self.symbol}/({other.symbol})")
        for unit in self.__slots__[1:]:
            setattr(new_units, unit, getattr(self, unit) - getattr(other, unit))
        
        # Return the new _Units object
        return new_units

    # Scalar power
    def __pow__(self, power: int | float) -> "_Units":

        # Check if the power is an int or float
        if not isinstance(power, (int, float)): raise ValueError()

        # Create a new _Units object
        new_units = _Units(f"({self.symbol})^{power}")
        for unit in self.__slots__[1:]:
            new_power = getattr(self, unit) * power
            if new_power == int(new_power):
                new_power = int(new_power)
            else:
                new_power = float(new_power)
            setattr(new_units, unit, new_power)
        
        # Return the new _Units object
        return new_units

    # Unit equality
    def __eq__(self, other: "_Units") -> bool:
            
        # Check if the other object is a _Units object
        if not isinstance(other, _Units): return False

        # Check if the units are equal
        for unit in self.__slots__[1:]:
            if getattr(self, unit) != getattr(other, unit): return False
        return True
    
    # Unit inequality
    def __ne__(self, other: "_Units") -> bool:
        return not (self == other)
    
    # Unit hash
    def __hash__(self) -> int:
        return hash((self.kg, self.m, self.s, self.A, self.K, self.mol, self.cd))







# User-friendly units class
class Units:
    kg  = _Units("kg", kg=1)
    m   = _Units("m", m=1)
    s   = _Units("s", s=1)
    A   = _Units("A", A=1)
    K   = _Units("K", K=1)
    mol = _Units("mol", mol=1)
    cd  = _Units("cd", cd=1)
    N   = _Units("N", kg=1, m=1, s=-2)
    J   = _Units("J", kg=1, m=2, s=-2)
    W   = _Units("W", kg=1, m=2, s=-3)
    C   = _Units("C", s=1, A=1)
    V   = _Units("V", kg=1, m=2, s=-3, A=-1)
    F   = _Units("F", kg=-1, m=-2, s=4, A=2)
    ohm = _Units("Ω", kg=1, m=2, s=-3, A=-2)
    S   = _Units("S", kg=-1, m=-2, s=3, A=2)
    Wb  = _Units("Wb", kg=1, m=2, s=-2, A=-1)
    T   = _Units("T", kg=1, s=-2, A=-1)
    H   = _Units("H", kg=1, m=2, s=-2, A=-2)
    lm  = _Units("lm", cd=1)
    lx  = _Units("lx", m=-2, cd=1)
    Gy  = _Units("Gy", m=2, s=-2)
    Sv  = _Units("Sv", m=2, s=-2)
    Hz  = _Units("Hz", s=-1)









class Datum:
    __slots__ = ["value", "uncertainty", "units"]

    # Define the Datum class
    def __init__(self, value: float, uncertainty: float, units: Union[str, _Units]) -> None:

        # Check the types of the inputs
        if not isinstance(value, (int, float)):       raise TypeError("Invalid type for value")
        if not isinstance(uncertainty, (int, float)): raise TypeError("Invalid type for uncertainty")
        if not isinstance(units, (str, _Units)):      raise TypeError("Invalid units")

        # Store the value, uncertainty, and units
        self.value = value
        self.uncertainty = uncertainty
        if isinstance(units, str):
            units, multiplicity = self.from_string(units)
            value *= multiplicity
            uncertainty *= multiplicity
        self.units = units

    # Define the __repr__ method for the Datum class
    def __repr__(self) -> str:
        return f"{self.value:.3f} ± {self.uncertainty:.3f} ({self.units})"
    
    # Define the operations for the Datum class
    def __add__(self, other: Union[int, float, "Datum"]) -> "Datum":

        # If instance of other is not Datum, int, or float, raise an error
        if not isinstance(other, (Datum, int, float)):
            raise TypeError("Invalid type for addition")
        
        # If other is an int or float, convert it to a Datum
        if isinstance(other, (int, float)):
            other = Datum(other, 0, "")
        
        # If the units are not the same, raise an error
        if self.units != other.units:
            raise UnitsError(f"Incompatible units ({self.units} & {other.units})")
        
        # Add the values and uncertainties
        new_value = self.value + other.value
        new_uncertainty = self.uncertainty + other.uncertainty
        return Datum(new_value, new_uncertainty, self.units)
    
    def __radd__(self, other: Union[int, float, "Datum"]) -> "Datum":
        return self + other

    def __sub__(self, other: Union[int, float, "Datum"]) -> "Datum":

        # If instance of other is not Datum, int, or float, raise an error
        if not isinstance(other, (Datum, int, float)):
            raise TypeError("Invalid type for subtraction")
        
        # If other is an int or float, convert it to a Datum
        if isinstance(other, (int, float)):
            other = Datum(other, 0, "")
        
        # If the units are not the same, raise an error
        if self.units != other.units:
            raise UnitsError(f"Incompatible units ({self.units} & {other.units})")
        
        # Subtract the values and add the uncertainties
        new_value = self.value - other.value
        new_uncertainty = self.uncertainty + other.uncertainty
        return Datum(new_value, new_uncertainty, self.units)

    def __rsub__(self, other: Union[int, float, "Datum"]) -> "Datum":
        return -1 * (self - other)
    
    def __mul__(self, other: Union[int, float, "Datum"]) -> "Datum":
        
        # If instance of other is not Datum, int, or float, raise an error
        if not isinstance(other, (Datum, int, float)):
            raise TypeError("Invalid type for multiplication")
        
        # If other is an int or float, convert it to a Datum
        if isinstance(other, (int, float)):
            if other == 0: return Datum(0, 0, "")
            other = Datum(other, 0, "")
        
        # Multiply the values and uncertainties
        new_value = self.value * other.value
        new_uncertainty = self.value * other.uncertainty + other.value * self.uncertainty
        return Datum(new_value, new_uncertainty, self.units)

    def __rmul__(self, other: Union[int, float, "Datum"]) -> "Datum":
        return self * other
    
    def __truediv__(self, other: Union[int, float, "Datum"]) -> "Datum":
            
        # If instance of other is not Datum, int, or float, raise an error
        if not isinstance(other, (Datum, int, float)):
            raise TypeError("Invalid type for division")
        
        # If other is an int or float, convert it to a Datum
        if isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError("Cannot divide by zero")
            other = Datum(other, 0, "")
        
        # Divide the values and uncertainties
        new_value = self.value / other.value
        new_uncertainty = (self.value * other.uncertainty + other.value * self.uncertainty) / other.value**2
        return Datum(new_value, new_uncertainty, self.units)
    
    def __rtruediv__(self, other: Union[int, float, "Datum"]) -> "Datum":
        return other / self

    def __pow__(self, power: Union[int, float]) -> "Datum":

        # Check if the power is an int or float
        if not isinstance(power, (int, float)): raise ValueError()
    
        # Power the values and uncertainties
        new_value = self.value ** power
        new_uncertainty = power * self.value ** (power-1) * self.uncertainty
        return Datum(new_value, new_uncertainty, self.units)

    def __eq__(self, other: "Datum") -> bool:
        if not isinstance(other, Datum): return False
        return self.value == other.value and self.uncertainty == other.uncertainty and self.units == other.units
    
    def __ne__(self, other: "Datum") -> bool:
        return not (self == other)
    
    def __hash__(self) -> int:
        return hash((self.value, self.uncertainty, self.units))

    # Parse a string into a Datum object
    def from_string(self, string: str) -> tuple[_Units, float]:  # Units, Multiplicity

        # If the string is empty, return Unitless
        if string == "": return _Units(""), 1

        # Split the string into a list of prefix-unit pairs
        pairs = string.split(" ")

        # Create a list of units and a multiplicity
        units_list = []
        multiplicity = 1
        for pair in pairs:
            unit, mult = self._from_string_pair(pair)
            units_list.append(unit)
            multiplicity *= mult
        
        # If there is only one unit, return it
        if len(units_list) == 1:
            return units_list[0], multiplicity
        
        # If there are multiple units, multiply them together
        else:
            new_units = units_list[0]
            for units in units_list[1:]:
                new_units *= units
            return new_units, multiplicity
        
    # Split a prefix-unit pair into a unit and a multiplicity
    @staticmethod
    def _from_string_pair(pair: str) -> tuple[_Units, float]:

        # If the pair contains a "^", split the pair into the unit and the power
        if "^" in pair:
            pair, power = pair.split("^")
            power = float(power)
        else:
            power = 1

        # If the pair is just a unit, return the unit and a multiplicity of 1
        if pair == "g": return _Units("kg", kg=power), 1e-3, 
        if pair in _Units.__dict__: return getattr(_Units, pair) ** power, 1

        # Otherwise find the unit and prefix
        for unit in _Units.__slots__[1:]:
            for prefix in [_ for _ in _Prefixes.__dict__.keys() if "_" not in _]:
                # print(pair, f"{prefix}{unit}")
                if pair == f"{prefix}{unit}":
                    _unit = Units.__dict__[unit]
                    _mult = _Prefixes.__dict__[prefix]
                    return _unit ** power, _mult ** power
        
        # Raise error if the pair is invalid
        raise ValueError("Invalid units")




# Write unit tests for the _Units class
def test_units():

    # Create a new _Units object
    units = _Units("J", kg=1, m=2, s=-2)

    # Test the __repr__ method
    if RAW_UNITS:
        assert units.__repr__() == "kg m^2 s^-2"
    else:
        assert units.__repr__() == "J"

    # Test the __mul__ method
    assert (units * units) == _Units("J^2", kg=2, m=4, s=-4)
    
    # Test the __truediv__ method
    assert (units / units) == _Units("Unitless")

    # Test the __pow__ method
    assert (units ** 2) == _Units("J^2", kg=2, m=4, s=-4)

    # Test the __eq__ method
    assert units == _Units("J", kg=1, m=2, s=-2)

    # Test the __ne__ method
    assert units != _Units("W", kg=1, m=2, s=-3)

    # Test the __hash__ method
    assert hash(units) == hash((1, 2, -2, 0, 0, 0, 0))

    # Print a success message
    print("All tests passed!")


# Run the unit tests if the script is run
if __name__ == "__main__":
    test_units()
                    
    


