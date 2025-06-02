import pymatgen.core as pmg
from chemlib import Element
import pandas as pd
import numpy as np
from nanodesclib.classes import *
from pymatgen.core import molecular_orbitals
from pathlib import Path
from thermo.chemical import Chemical
from itertools import chain
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from pymatgen.analysis.local_env import CrystalNN

current_path = Path(__file__).parent.resolve()
csv_file_path = current_path / "reference/polarizabilities.csv"
polarizabilities = pd.read_csv(csv_file_path)


class Descriptors:

    def __init__(self, formula, smiles=None, structure=None):
        self.formula = formula
        self.smiles = smiles
        self.structure = structure
        self.compound_class = assign_class(formula)
        try:
            if self.compound_class._type not in ['composite', 'coreshell']:
                self.parts = [formula]
            else:
                self.parts = self.compound_class.consist()
        except:
            self.parts = [formula]

    def _get_composition(self, formula):
        return pmg.Composition(formula)

    def _apply(self, func):
        results = [func(p) for p in self.parts]
        if len(results) == 1:
            return results[0]
        return sum(results)

    def number_of_atoms(self):
        return self._apply(lambda f: self._get_composition(f).num_atoms)

    def molecular_weight(self):
        return self._apply(lambda f: self._get_composition(f).weight)

    def average_electronegativity(self):
        return self._apply(lambda f: self._get_composition(f).average_electroneg)

    def average_electron_affinity(self):
        def calc(formula):
            comp = self._get_composition(formula)
            elements = comp.as_dict()
            result = 0
            for el in elements:
                e = pmg.Element(el)
                weight_fraction = comp.get_wt_fraction(e)
                result += (e.electron_affinity or 0) * weight_fraction
            return result
        return self._apply(calc)

    def polarizability(self):
        def calc(formula):
            comp = self._get_composition(formula)
            elements = comp.as_dict()
            result = 0
            for el in elements:
                pol_row = polarizabilities[polarizabilities['Atom'] == el]
                if not pol_row.empty:
                    try:
                        val = float(pol_row['αD'].values[0].split(' ± ')[0])
                        wt = comp.get_wt_fraction(pmg.Element(el))
                        result += val * wt
                    except:
                        pass
            return result
        return self._apply(calc)

    def homo_lumo(self):
        try:
            mo = pmg.molecular_orbitals.MolecularOrbitals(self.formula).band_edges
            homo = mo['HOMO'][-1]
            lumo = mo['LUMO'][-1]
            hardness = (lumo - homo) / 2
            softness = 1 / (2 * hardness)
            electrophilicity = (hardness * ((-homo) ** 2)) / (2 * hardness)
            chemical_potential = -0.5 * (homo + lumo)
            return {
                'HOMO': homo,
                'LUMO': lumo,
                'absolute_hardness': hardness,
                'softness': softness,
                'electrophilicity_index': electrophilicity,
                'chemical_potential': chemical_potential
            }
        except:
            return {
                'HOMO': None,
                'LUMO': None,
                'absolute_hardness': None,
                'softness': None,
                'electrophilicity_index': None,
                'chemical_potential': None
            }

    def smiles_descriptors(self):
        if self.smiles:
            smiles_list = self.smiles if isinstance(self.smiles, list) else [self.smiles]
            result = {}
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    desc = Descriptors.CalcMolDescriptors(mol)
                    result.update({f"smiles_{i}_{k}": v for k, v in desc.items()})
                    # Добавим геометрические, которых нет в CalcMolDescriptors
                    try:
                        result[f"smiles_{i}_LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
                        result[f"smiles_{i}_NPR1"] = rdMolDescriptors.CalcNPR1(mol)
                        result[f"smiles_{i}_NPR2"] = rdMolDescriptors.CalcNPR2(mol)
                    except:
                        continue
            return result
        return {}

    def thermo_descriptors(self):
        useful_keys = {
            'Tboil', 'Tmelting', 'Tc', 'Pc', 'Vc', 'Zc', 'omega', 'Hf', 'S0g', 'GWP', 'ODP',
            'logP', 'RI', 'dipole', 'enthalpy_of_formation', 'critical_temperature',
            'thermal_conductivity', 'density', 'surface_tension', 'viscosity', 'heat_capacity'
        }

        def extract_thermo(formula):
            try:
                d = Chemical(formula).__dict__
                return {f"{formula}_{k}": v for k, v in d.items() if k in useful_keys and isinstance(v, (int, float))}
            except:
                return {}

        result = {}
        for part in self.parts:
            result.update(extract_thermo(part))
        return result

    def electronic_descriptors(self):
        def calc(formula):
            comp = self._get_composition(formula)
            elements = comp.as_dict()
            total_ve = 0
            s, p, d, f = 0, 0, 0, 0
            for el, amt in elements.items():
                e = pmg.Element(el)
                conf = e.full_electronic_structure
                total_ve += e.full_electronic_structure[-1][2] * amt
                for orb in conf:
                    subshell = orb[1]
                    if subshell == 0:
                        s += orb[2] * amt
                    elif subshell == 1:
                        p += orb[2] * amt
                    elif subshell == 2:
                        d += orb[2] * amt
                    elif subshell == 3:
                        f += orb[2] * amt
            return {
                f"{formula}_valence_electrons": total_ve,
                f"{formula}_s_electrons": s,
                f"{formula}_p_electrons": p,
                f"{formula}_d_electrons": d,
                f"{formula}_f_electrons": f
            }

        result = {}
        for part in self.parts:
            result.update(calc(part))
        return result

    def structural_descriptors(self):
        result = {}
        if self.structure:
            try:
                result['structure_volume'] = self.structure.volume
                result['structure_density'] = self.structure.density
                cnn = CrystalNN()
                coordination_numbers = []
                for site in self.structure:
                    try:
                        neighbors = cnn.get_nn_info(self.structure, self.structure.index(site))
                        coordination_numbers.append(len(neighbors))
                    except:
                        continue
                if coordination_numbers:
                    result['average_coordination_number'] = np.mean(coordination_numbers)
            except:
                pass
        return result

    def all_descriptors(self):
        desc = {
            'number_of_atoms': self.number_of_atoms(),
            'molecular_weight': self.molecular_weight(),
            'average_electronegativity': self.average_electronegativity(),
            'average_electron_affinity': self.average_electron_affinity(),
            'polarizability': self.polarizability()
        }
        desc.update(self.homo_lumo())
        desc.update(self.thermo_descriptors())
        desc.update(self.smiles_descriptors())
        desc.update(self.electronic_descriptors())
        desc.update(self.structural_descriptors())
        return desc
