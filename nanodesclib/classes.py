import pymatgen.core as pmg
import re
from nanodesclib.assign_class_patterns import *


def assign_class(text):
    rules = [
        (comp, Composite),
        (metal, Metal),
        (metal_ox, MetalOxide),
        (salt, Salt),
        (mix_ox, ComplexOxide),
        (bime, BiMetal),
        (mix_salt, ComplexSalt),
        (trme, TriMetal),
        (me_hydrox, MetalHydroxide),
        (nonmetal, NonMetal),
        (carbide, Carbide),
        (qme, TetraMetal),
        (nitride, Nitride),
        (phosphide, Phosphide),
        (nonmetal_compound, NonmetalCompound),
        (pentame, PentaMetal)
    ]

    for pattern, cls in rules:
        if re.fullmatch(pattern, text.strip()):
            return cls(text.strip())

    return 'Other'


class Metal:
    _type = 'metal'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class MetalOxide:
    _type = 'metal_oxide'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class Salt:
    _type = 'salt'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class NonMetal:
    _type = 'non_metal'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class Carbide:
    _type = 'carbide'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class Nitride:
    _type = 'nitride'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class Phosphide:
    _type = 'phosphide'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class NonmetalCompound:
    _type = 'nonmetal_compound'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class CoreShell:
    def __init__(self, formula):
        self._type = 'coreshell'
        self.formula = formula

    def consist(self):
        composition = dict()
        c = 1
        content = self.formula.split('@')
        composition['core'] = assign_class(content[0]).consist()
        for i in content[1:]:
            composition['shell_'+str(c)] = assign_class(i).consist()
            c += 1
        return composition


class Composite(CoreShell):
    _type = 'composite'

    def __init__(self, formula):
        self.formula = formula
        if '@' in formula:
            super().__init__(formula)

    def consist(self):
        if '@' in self.formula:
            composition = super().consist()
        else:
            composition = dict()
            c = 1
            if '/' in self.formula:
                content = self.formula.split('/')
                for i in content:
                    composition['compound_'+str(c)] = assign_class(i).consist()
                    c += 1
            elif '-' in self.formula:
                content = self.formula.split('-')
                for i in content:
                    composition['compound_'+str(c)] = assign_class(i).consist()
                    c += 1
            else:
                content = self.formula.split('–')
                for i in content:
                    composition['compound_'+str(c)] = assign_class(i).consist()
                    c += 1
        return composition


class ComplexOxide:
    _type = 'complex_metal_oxide'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class ComplexSalt:
    _type = 'complex_salt'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class MetalHydroxide:
    _type = 'metal_hydroxide'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class BiMetal:
    _type = 'bimetal'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class TriMetal:
    _type = 'trimetal'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class TetraMetal:
    _type = 'tetrametal'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula


class PentaMetal:
    _type = 'pentametal'

    def __init__(self, formula):
        self.formula = formula
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.formula
