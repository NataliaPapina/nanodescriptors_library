import pymatgen.core as pmg
import re
from nanodesclib.assign_class_patterns import *
from pymatgen.core import Composition
from itertools import chain

def normalize_formula(formula):

    if not formula or not isinstance(formula, str):
        return []

    formula = formula.replace('\xa0', '').replace('–', '-').replace('−', '-').strip()

    parts = re.split(r'[@\-/]', formula)

    cleaned = []
    for part in parts:
        part = part.strip()
        try:
            if len(re.split(r'[@\-/]', part)) < 2:
                cleaned.append(part)
        except:
            continue

    return cleaned


def assign_class(text):
    rules = [
        (comp, Composite),
        (metal, Metal),
        (metallate, Metallate),
        (metal_ox, MetalOxide),
        (mix_ox, ComplexOxide),
        #(bime, BiMetal),
        #(trme, TriMetal),
        (me_hydrox, MetalHydroxide),
        (nonmetal, NonMetal),
        (carbide_halogenide, Carbide),
        (intermetallic_carbide, Intermetallic),
        (carbide, Carbide),
        #(qme, TetraMetal),
        #(nitride, Nitride),
        #(phosphide, Phosphide),
        (nonmetal_compound, NonmetalCompound),
        #(pentame, PentaMetal),
        (intermetallic, Intermetallic),
        (me_pnictide, MetalPnictide),
        (borocarbide, BoroCarbide),
        (boronitride, BoroNitride),
        (kesterite, Kesterite),
        (complex_chalcogenide, ComplexChalcogenide),
        (carbonate, Salt),
        (salt, Salt),
        (thiophosphate, ComplexSalt),
        (selenophosphate, ComplexSalt),
        (selenide_halide, ComplexSalt),
        (chalcogen_halide, ComplexSalt),
        (complex_halogenide, ComplexSalt),
        (mix_salt, ComplexSalt),
    ]

    for pattern, cls in rules:
        if re.fullmatch(pattern, text.strip()):
            return cls(text.strip())

    return 'Other'


class Metal:
    _type = 'metal'

    def __init__(self, formula):
        self.formula = formula
        self.core = self.formula

    def consist(self):
        return [self.formula]

class BoroNitride:
    _type = 'boronitride'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class Kesterite:
    _type = 'kesterite'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class ComplexChalcogenide:
    _type = 'complex_chalcogenide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]

class BoroCarbide:
    _type = 'borocarbide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]

class MetalOxide:
    _type = 'metal_oxide'

    def __init__(self, formula):
        self.formula = formula
        self.core = self.formula

    def consist(self):
        return [self.formula]

class MetalPnictide:
    _type = 'metal_pnictide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]

class Salt:
    _type = 'salt'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class NonMetal:
    _type = 'non_metal'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class Carbide:
    _type = 'carbide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class Nitride:
    _type = 'nitride'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class Phosphide:
    _type = 'phosphide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class NonmetalCompound:
    _type = 'nonmetal_compound'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class CoreShell:
    def __init__(self, formula):
        self._type = 'coreshell'
        self.formula = formula

    def consist(self):
        content = self.formula.split('@')
        return list(chain.from_iterable(normalize_formula(x) for x in content))


class Composite(CoreShell):
    _type = 'composite'

    def __init__(self, formula):
        self.formula = formula
        if '@' in formula:
            super().__init__(formula)

    def consist(self):
        if '@' in self.formula:
            return super().consist()

        if '/' in self.formula:
            content = self.formula.split('/')
        elif '-' in self.formula:
            content = self.formula.split('-')
        else:
            content = self.formula.split('–')

        return list(chain.from_iterable(normalize_formula(x) for x in content))


class ComplexOxide:
    _type = 'complex_metal_oxide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class ComplexSalt:
    _type = 'complex_salt'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class MetalHydroxide:
    _type = 'metal_hydroxide'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]


class BiMetal:
    _type = 'bimetal'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return self.formula


class TriMetal:
    _type = 'trimetal'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return self.formula


class TetraMetal:
    _type = 'tetrametal'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return self.formula


class PentaMetal:
    _type = 'pentametal'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return self.formula

class Intermetallic:
    _type = 'intermetallic'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]

class Metallate:
    _type = 'metallate'

    def __init__(self, formula):
        self.formula = formula

    def consist(self):
        return [self.formula]