import re
import pymatgen.core as pmg


Alkali_Metals = 'Li|Na|K|Rb|Cs|Fr'
Alkaline_Earth_Metals = 'Be|Mg|Ca|Sr|Ba|Ra'
Transition_Metals = 'La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Sc|Y|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn'
Basic_Metals = 'Al|Ga|In|Sn|Tl|Pb|Bi|Nh|Fl|Mc|Lv|Po'
Metalloids = 'B|Si|Ge|As|Sb|Te|At|Ts'

All_metals = '|'.join([Alkali_Metals, Alkaline_Earth_Metals, Transition_Metals, Basic_Metals, Metalloids])

Halogens = 'F|Cl|Br|I'
Chalcogen = 'O|S|Se'
Pnictogens = 'N|P'
Carbon = 'C'
Hydrogen = 'H'

Non_metals = '|'.join([Halogens, Chalcogen, Pnictogens, Carbon, Hydrogen, Metalloids])

Noble_gases = 'He|Ne|Ar|Kr|Xe|Rn'

num = '([0-9]{0,2}[.,]?[0-9]{0,})?'

nonmetal = f'(F|Cl|Br|I|O|S|Se|N|P|C|H|B|Si|Ge|As|Sb|Te|At|Ts){num}'

nonmetal_compound = nonmetal*2 + ('(' + nonmetal + ')?')*2

anion = '(' + '|'.join(['(' + nonmetal + 'O' + num + ')', '(' + r'(F|Cl|Br|I|S|Se|B|Si|Ge|As|Sb|Te|At|Ts|\(?PW12O40\)?|\(?SiW12O40\)?|\(?CN\)?|\(?PW11O39\)?)[0-9]{0,2}' + ')', '(' + nonmetal + num + nonmetal + num + 'O' + num + ')']) + num + ')'

metal = r'\(?' + f'({All_metals})' + num + r'\)?' + num

metal_ox = r'\(?' + f'({All_metals})' + num + 'O' + num + r'\)?' + num

salt = r'\(?' + f'({All_metals})' + num + r'\(?' + '(O)?' + '(OH)?' + r'\)?' + num + r'\(?' + anion + r'\)?' + num + r'\(?' + '(O)?' + r'\)?' + num + r'\(?' + '(OH)?' + r'\)?' + num + r'\)?'

mix_ox = r'\(?' + f'({All_metals})' + num + 'O?' + num + r'\)?' + (r'(\(?' + f'({All_metals})' + num + r'\)?)?')*5 + r'\(?' + f'({All_metals})' + num + 'O' + num + r'\)?' + num + r'\)?'

me_hydrox = r'\(?' + f'({All_metals})' + num + '(O)?' + num + r'\(?' + 'OH' + r'\)?' + num + r'\)?'

mix_salt = r'\(?' + f'({All_metals})' + r'\)?' + num + r'\(?' + (f'({All_metals})?' + num + anion + '?' + num)*2 + anion + num + f'({All_metals})?' + num + (anion + '?' + num)*4 + num + r'\)?' + num + 'O?' + num

bime = r'\(?' + (f'({All_metals})' + num)*2 + r'\)?' + num

trme = r'\(?' + (f'({All_metals})' + num)*3 + r'\)?' + num

qme = r'\(?' + (f'({All_metals})' + num)*4 + r'\)?' + num

pentame = r'\(?' + (f'({All_metals})' + num)*5 + r'\)?' + num

carbide = r'\(?' + f'({All_metals})' + num + 'C' + num + r'\)?'

nitride = r'\(?' + f'({All_metals})' + num + 'N' + num + r'\)?'
phosphide = r'\(?' + f'({All_metals})' + num + 'P' + num + r'\)?'


class Metal:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'metal'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class MetalOxide:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'metal_oxide'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class Salt:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'salt'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class NonMetal:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'non_metal'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class Carbide:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'carbide'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class Nitride:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'nitride'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class Phosphide:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'phosphide'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class NonmetalCompound:
     def __init__(self, formula):
        self.formula = formula
        self._type = 'nonmetal_compound'
        self.core = pmg.Composition(self.formula)

     def consist(self):
        return self.core


class CoreShell:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'coreshell'

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
    def __init__(self, formula):
        self.formula = formula
        self._type = 'composite'
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
    def __init__(self, formula):
        self.formula = formula
        self._type = 'complex_metal_oxide'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class ComplexSalt:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'complex_salt'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class MetalHydroxide:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'metal_hydroxide'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class BiMetal:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'bimetal'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class TriMetal:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'trimetal'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class TetraMetal:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'tetrametal'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


class PentaMetal:
    def __init__(self, formula):
        self.formula = formula
        self._type = 'pentametal'
        self.core = pmg.Composition(self.formula)

    def consist(self):
        return self.core


comp = r'[\(\)]?\w+[.,\(\)]?([\w.,\(\)]?)+(-|/|–|@)[\(\)]?\w+[.,\(\)]?([\w.,\(\)]?)+(((-|/|–|@)[\(\)]?\w+[.,\(\)]?([\w.,\(\)]?)+)?)+\b'


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

