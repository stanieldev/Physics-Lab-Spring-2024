from enum import Enum
import re
from decimal import Decimal


# Make a function that converts a string to a UnitsBinary object
# Example: "kg m^2 s^-2" -> UnitsBinary(kg=1, m=2, s=-2)
def string_to_units(string: str) -> tuple["UnitsBinary", float]:
    if string == "": return UnitsBinary("Unitless"), 1
    units_list = []
    multiplicity = 1
    for combo in string.split(" "):

        # If there is a exponent, split the string into the chars and the power
        power_str = "1"
        if "^" in combo:
            combo, power_str = combo.split("^")
        
        # If combo is just a unit, add it to the list
        if combo in Units.__dict__:
            units_list.append(getattr(Units, combo) ** int(power_str))
            continue

        # If it is just grams, add it to the list
        if combo == "g":
            units_list.append(Units.kg ** int(power_str))
            multiplicity *= 1e-3 ** int(power_str)
            continue

        # Find a valid prefix + unit combination
        match = re.match(r"([a-zA-Z]+)([a-zA-Z]+)", combo)

        # If there is no match, raise an error
        if match is None:
            raise ValueError("Invalid units")
        
        # Get the prefix and unit
        prefix_str, unit_str = match.groups()

        # If the prefix is not in the Prefixes class, raise an error
        if prefix_str not in Prefixes.__dict__:
            raise ValueError("Invalid prefix")
        
        # Convert the prefix to a float based on the Prefixes class
        prefix = Prefixes.__dict__[prefix_str]

        # Multiply the value and uncertainty by the prefix
        multiplicity *= prefix ** int(power_str)

        # Return the units as a UnitsBinary object
        component_units = getattr(Units, unit_str) ** int(power_str)
        units_list.append(component_units)
    
    # If there is only one unit, return it
    if len(units_list) == 1:
        return units_list[0], multiplicity
    
    # If there are multiple units, multiply them together
    else:
        new_units = units_list[0]
        for units in units_list[1:]:
            new_units *= units
        return new_units, multiplicity



# Define Units in terms of the SI units
class UnitsBinary:
    __slots__ = ["symbol", "kg", "m", "s", "A", "K", "mol", "cd"]

    def __init__(self, symbol: str = None, kg=0, m=0, s=0, A=0, K=0, mol=0, cd=0) -> None:
        self.symbol = symbol
        self.kg = kg
        self.m = m
        self.s = s
        self.A = A
        self.K = K
        self.mol = mol
        self.cd = cd

    def _style_1(self) -> str:
        repr_front = []
        repr_back = []
        for unit in self.__slots__[1:]:
            power = getattr(self, unit)
            if power == 1: repr_front.append(f"{unit}")
            elif power > 0: repr_front.append(f"{unit}^{power}")
            elif power < 0: repr_back.append(f"{unit}^{power}")
        if len(repr_front) + len(repr_back) == 0: return "Unitless"
        else: return " ".join(repr_front+repr_back)

    def _style_2(self) -> str:  # WIP
        repr_front = ""
        repr_back = ""
        for unit in self.__slots__[1:]:
            power = getattr(self, unit)
            if power == 1: repr_front += f"{unit} "
            elif power > 0: repr_front += f"{unit}^{power} "
            elif power < 0: repr_back += f"{unit}^{-power} "
        if repr_front + repr_back == "": return "Unitless"
        elif repr_back == "": return repr_front
        elif repr_front == "": return f"1 / {repr_back}"
        else: return f"{repr_front}/ {repr_back}"
    
    def __repr__(self) -> str:
        for unit in [item for item in Units.__dict__ if not item.startswith("_")]:
            if getattr(Units, unit) == self:
                return unit

        return self._style_1()


    # Define the operations for the UnitsBinary class
    def __mul__(self, other: "UnitsBinary") -> "UnitsBinary":
        new_unit_object = UnitsBinary(f"{self.symbol} {other.symbol}")
        for unit in self.__slots__[1:]:
            setattr(new_unit_object, unit, getattr(self, unit) + getattr(other, unit))
        return new_unit_object
    
    def __rmul__(self, other: "UnitsBinary") -> "UnitsBinary":
        return self * other

    def __truediv__(self, other: "UnitsBinary") -> "UnitsBinary":
        new_unit_object = UnitsBinary(f"{self.symbol} ({other.symbol})^-1")
        for unit in self.__slots__[1:]:
            setattr(new_unit_object, unit, getattr(self, unit) - getattr(other, unit))
        return new_unit_object
    
    def __rtruediv__(self, other: "UnitsBinary") -> "UnitsBinary":
        if isinstance(other, (int, float)):
            new_units = UnitsBinary(f"{self.symbol}^-1")
            for unit in self.__slots__[1:]:
                setattr(new_units, unit, -getattr(self, unit))
            return new_units
        return other / self

    def __pow__(self, power: int):
        new_units = UnitsBinary(f"{self.symbol}^{power}")
        for unit in self.__slots__[1:]:
            new_power = getattr(self, unit) * power
            if new_power == int(new_power):
                new_power = int(new_power)
            else:
                new_power = float(new_power)
            setattr(new_units, unit, new_power)
        return new_units
    

    # Define the comparison operations for the UnitsBinary class
    def __eq__(self, other: "UnitsBinary") -> bool:
        for unit in self.__slots__[1:]:
            if getattr(self, unit) != getattr(other, unit):
                return False
        return True
    
    def __ne__(self, other: "UnitsBinary") -> bool:
        return not (self == other)
    
    def __hash__(self) -> int:
        return hash((self.kg, self.m, self.s, self.A, self.K, self.mol, self.cd))

class Prefixes:
    Y = 1e24
    Z = 1e21
    E = 1e18
    P = 1e15
    T = 1e12
    G = 1e9
    M = 1e6
    k = 1e3
    h = 1e2
    da = 1e1
    d = 1e-1
    c = 1e-2
    m = 1e-3
    u = 1e-6
    n = 1e-9
    p = 1e-12
    f = 1e-15
    a = 1e-18
    z = 1e-21
    y = 1e-24

class Units:
    kg = UnitsBinary("kg", kg=1)
    m = UnitsBinary("m", m=1)
    s = UnitsBinary("s", s=1)
    A = UnitsBinary("A", A=1)
    K = UnitsBinary("K", K=1)
    mol = UnitsBinary("mol", mol=1)
    cd = UnitsBinary("cd", cd=1)
    N = UnitsBinary("N", kg=1, m=1, s=-2)
    J = UnitsBinary("J", kg=1, m=2, s=-2)
    W = UnitsBinary("W", kg=1, m=2, s=-3)
    C = UnitsBinary("C", s=1, A=1)
    V = UnitsBinary("V", kg=1, m=2, s=-3, A=-1)
    F = UnitsBinary("F", kg=-1, m=-2, s=4, A=2)
    ohm = UnitsBinary("Ω", kg=1, m=2, s=-3, A=-2)
    S = UnitsBinary("S", kg=-1, m=-2, s=3, A=2)
    Wb = UnitsBinary("Wb", kg=1, m=2, s=-2, A=-1)
    T = UnitsBinary("T", kg=1, s=-2, A=-1)
    H = UnitsBinary("H", kg=1, m=2, s=-2, A=-2)
    lm = UnitsBinary("lm", cd=1)
    lx = UnitsBinary("lx", m=-2, cd=1)
    Gy = UnitsBinary("Gy", m=2, s=-2)
    Sv = UnitsBinary("Sv", m=2, s=-2)
    Hz = UnitsBinary("Hz", s=-1)



# Define a class called Datum that will hold a value, uncertainty, and units
class Datum:
    def __init__(self, value: float, uncertainty: float, units: str | UnitsBinary) -> None:
        self.units, _multiplicity = self._read_units(units)
        self.value = _multiplicity * value
        self.uncertainty = _multiplicity * uncertainty
    
    def _read_units(self, units: str | UnitsBinary) -> UnitsBinary:
        if isinstance(units, UnitsBinary):
            return units, 1
        elif isinstance(units, str):
            return string_to_units(units)
        else:
            raise ValueError("Invalid units")

    def __repr__(self) -> str:
        value = Decimal(str(self.value)).normalize()
        uncertainty = Decimal(str(self.uncertainty)).normalize()
        return f"{value} ± {uncertainty} ({self.units})"
    
    def __add__(self, other: "Datum") -> "Datum":
        
        # Cast other to a Datum if it is not already
        if not isinstance(other, Datum):
            other = Datum(other, 0, "")

        # If the units are not the same, raise an error
        if self.units != other.units:
            raise ValueError("Incompatible units")
        
        # Add the values and uncertainties
        new_value = self.value + other.value
        new_uncertainty = self.uncertainty + other.uncertainty
        return Datum(new_value, new_uncertainty, self.units)
    
    def __radd__(self, other: "Datum") -> "Datum":
        return self + other

    def __sub__(self, other: "Datum") -> "Datum":

        # Cast other to a Datum if it is not already
        if not isinstance(other, Datum):
            other = Datum(other, 0, "")

        # If the units are not the same, raise an error
        if self.units != other.units:
            raise ValueError("Incompatible units")
        
        # Add the values and uncertainties
        new_value = self.value - other.value
        new_uncertainty = self.uncertainty + other.uncertainty
        return Datum(new_value, new_uncertainty, self.units)
    
    def __rsub__(self, other: "Datum") -> "Datum":
        return -1 * (self - other)
    
    def __mul__(self, other) -> "Datum":
        if isinstance(other, Datum):
            new_value = self.value * other.value
            new_uncertainty = self.value * other.uncertainty + other.value * self.uncertainty
            new_units = self.units * other.units
            return Datum(new_value, new_uncertainty, new_units)
        else:
            new_value = self.value * other
            new_uncertainty = self.uncertainty * other
            return Datum(new_value, new_uncertainty, self.units)
    
    def __rmul__(self, other) -> "Datum":
        return self * other

    def __truediv__(self, other) -> "Datum":
        if isinstance(other, Datum):
            new_value = self.value / other.value
            new_uncertainty = (self.value * other.uncertainty + other.value * self.uncertainty) / other.value**2
            new_units = self.units / other.units
            return Datum(new_value, new_uncertainty, new_units)
        else:
            new_value = self.value / other
            new_uncertainty = self.uncertainty / other
            return Datum(new_value, new_uncertainty, self.units)
    
    def __rtruediv__(self, other) -> "Datum":
        if isinstance(other, Datum):
            return other / self
        else:
            new_value = other / self.value
            new_uncertainty = other * self.uncertainty / self.value ** 2
            new_units = 1 / self.units
            return Datum(new_value, new_uncertainty, new_units)

    def __pow__(self, power: int) -> "Datum":
        new_value = self.value ** power
        new_uncertainty = power * self.value ** (power-1) * self.uncertainty
        new_units = self.units ** power
        return Datum(new_value, new_uncertainty, new_units)

