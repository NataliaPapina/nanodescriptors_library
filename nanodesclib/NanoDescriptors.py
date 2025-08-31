import pymatgen.core as pmg
from chemlib import Element
import pubchempy as pcp
import inspect
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
from nanodesclib.aflow import AflowDescriptors
import numbers
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

current_path = Path(__file__).parent.resolve()
csv_file_path = current_path / "reference/polarizabilities.csv"
polarizabilities = pd.read_csv(csv_file_path)


class NanoDescriptors:
    cas_cache = {}
    thermo_cache = {}

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

                    try:
                        result[f"smiles_{i}_LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
                        result[f"smiles_{i}_NPR1"] = rdMolDescriptors.CalcNPR1(mol)
                        result[f"smiles_{i}_NPR2"] = rdMolDescriptors.CalcNPR2(mol)
                    except:
                        continue
            return result
        return {}

    def atomic_mechanical_descriptors(self):
        descriptor_names = [
            'atomic_number', 'atomic_mass', 'electron_affinity', 'first_ionization_energy',
            'atomic_radius_calculated', 'van_der_waals_radius', 'electrical_resistivity',
            'velocity_of_sound', 'reflectivity', 'refractive_index', 'poissons_ratio',
            'molar_volume', 'thermal_conductivity', 'boiling_point', 'melting_point',
            'critical_temperature', 'superconduction_temperature', 'liquid_range',
            'bulk_modulus', 'youngs_modulus', 'brinell_hardness', 'rigidity_modulus',
            'mineral_hardness', 'vickers_hardness', 'density_of_solid',
            'coefficient_of_linear_thermal_expansion'
        ]

        descriptor_sums = {name: 0.0 for name in descriptor_names}
        total_weight = 0.0

        for formula in self.parts:
            comp = self._get_composition(formula)
            for el_symbol, amt in comp.items():
                try:
                    el = pmg.Element(el_symbol)
                    weight_frac = comp.get_wt_fraction(el)
                    total_weight += weight_frac
                    for name in descriptor_names:
                        val = getattr(el, name, None)
                        if isinstance(val, numbers.Number):
                            descriptor_sums[name] += val * weight_frac
                except Exception as e:
                    print(f"[WARN] Failed descriptor for element {el_symbol}: {e}")
                    continue

        averaged = {
            f"avg_{name}": (val / total_weight if total_weight else None)
            for name, val in descriptor_sums.items()
        }
        return averaged

    def get_thermo_descriptors(self):
        formulas = self.parts

        try:
            weights = [Composition(f).weight for f in formulas]
            total_weight = sum(weights)
            weight_fractions = [w / total_weight for w in weights]
        except Exception:
            weights = [1.0 for _ in formulas]
            weight_fractions = [1.0 / len(formulas)] * len(formulas)

        skip_keys = {
            'ChemicalMetadata', 'synonyms', 'InChI', 'CAS', 'autocalc', 'PubChem', 'formula', 'MW', 'atoms',
            'eos_in_a_box', 'Tm_sources', 'Tm_source', 'Tflash_source', 'Tautoignition_sources', 'Pt_sources',
            'Tautoignition_source', 'TWA_sources', 'TWA_source', 'STEL_sources', 'STEL_source', 'Pt_source',
            'Hfus_methods', 'Hfus_method', 'Tb_sources', 'Tb_source', 'Tc_methods', 'Tc_method', 'Pc_methods',
            'Vc_methods', 'Vc_method', 'omega_methods', 'Carcinogen', 'Pc_method', 'InChI_Key', 'IUPAC_name',
            'ID', 'omega_method', 'Tt_sources', 'Tflash_sources', 'Stockmayer_sources', 'Tt_source',
            'Ceiling_sources', 'Ceiling_source', 'Skin_sources', 'Skin_source', 'Hfg_sources', 'Hfg_source',
            'S0g_sources', 'S0g_source', 'dipole_sources', 'dipole_source', 'GWP_sources', 'GWP_source',
            'ODP_sources', 'ODP_source', 'logP_sources', 'logP_source', 'RI_sources', 'RI_source',
            'conductivity_sources', 'conductivity_source', 'name', 'Stockmayer_source', 'VaporPressure',
            'Hf_sources', 'combustion_stoichiometry', 'elemental_reaction_data', 'VolumeGas', 'VolumeLiquid',
            'VolumeSolid', 'HeatCapacityGas', 'HeatCapacitySolid', 'HeatCapacityLiquid', 'EnthalpyVaporization',
            'EnthalpySublimation', 'PermittivityLiquid', 'Permittivity', 'SurfaceTension',
            'ThermalConductivitySolid', 'ThermalConductivityGas', 'ViscosityGas', 'ThermalConductivityLiquid',
            'SublimationPressure', 'ViscosityLiquid', '_eos_T_101325', 'eos_Tb', 'molecular_diameter_sources',
            'molecular_diameter_source', 'LFL_sources', 'LFL_source', 'UFL_sources', 'UFL_source'
        }

        all_keys = set()
        part_descriptors = []

        for formula, weight_fraction in zip(formulas, weight_fractions):
            try:
                chem = Chemical(formula)
            except Exception:
                continue

            descriptors = {}
            for attr in dir(chem):
                if attr.startswith('_') or attr in skip_keys:
                    continue
                try:
                    value = getattr(chem, attr)
                    if callable(value):
                        sig = inspect.signature(value)
                        if len(sig.parameters) > 0:
                            continue
                        value = value()
                    if isinstance(value, (dict, list, tuple)):
                        continue
                    if isinstance(value, (int, float, str, bool, np.number)):
                        descriptors[attr] = value
                except Exception:
                    continue

            descriptors['weight_fraction'] = weight_fraction
            part_descriptors.append(descriptors)
            all_keys.update(descriptors.keys())

        result = {}
        for key in all_keys:
            values = []
            weights = []
            for d in part_descriptors:
                if key in d and isinstance(d[key], (int, float)):
                    values.append(d[key])
                    weights.append(d['weight_fraction'])
            if values:
                result['thermo_' + key] = np.average(values, weights=weights)

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

    def aflow_descriptors(self):
        formulas = self.parts
        descriptors_list = []
        weights = []

        for formula in formulas:
            try:
                weight = Composition(formula).weight
                aflow = AflowDescriptors(formula)
                aflow.fetch()
                desc = aflow.get_descriptors()
                if desc:
                    descriptors_list.append(desc)
                    weights.append(weight)
            except Exception as e:
                print(f"[AFLOW WARN] {formula}: {e}")
                continue

        if not descriptors_list:
            return {}, {}

        # нормализация весов
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        all_keys = set(k for d in descriptors_list for k in d.keys())
        averaged = {}

        for key in all_keys:
            values = []
            ws = []
            for d, w in zip(descriptors_list, weights):
                val = d.get(key, None)
                if isinstance(val, (int, float)) and not pd.isna(val):
                    values.append(val)
                    ws.append(w)
            if values:
                averaged["aflow_" + key] = np.average(values, weights=ws)
            else:
                # для строковых/категориальных признаков сохраним список
                cat_vals = [d.get(key) for d in descriptors_list if isinstance(d.get(key), str)]
                if cat_vals:
                    averaged["aflow_" + key] = list(set(cat_vals))  # или cat_vals[0] (первое значение)

        return averaged, descriptors_list

    def all_descriptors(self):
        desc = {
            'number_of_atoms': self.number_of_atoms(),
            'molecular_weight': self.molecular_weight(),
            'average_electronegativity': self.average_electronegativity(),
            'average_electron_affinity': self.average_electron_affinity(),
            'polarizability': self.polarizability()
        }
        desc.update(self.homo_lumo())
        desc.update(self.get_thermo_descriptors())
        desc.update(self.smiles_descriptors())
        desc.update(self.electronic_descriptors())
        desc.update(self.structural_descriptors())
        desc.update(self.atomic_mechanical_descriptors())
        aflow_desc, _ = self.aflow_descriptors()
        desc.update(aflow_desc)
        return desc
