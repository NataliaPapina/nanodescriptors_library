import pymatgen.core as pmg
from chemlib import Element
import pubchempy as pcp
import inspect
import pandas as pd
import numpy as np
from nanodesclib.classes import *
from nanodesclib.ElementDescriptors import *
from nanodesclib.wt_fraction import *
from nanodesclib.el_amt_dict import *
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
        inorganic_formula, organic_smiles = self.extract_organic_components(formula)

        if inorganic_formula:
            self.formula = inorganic_formula
        else:
            self.formula = formula
        all_smiles = []
        if smiles:
            if isinstance(smiles, list):
                all_smiles.extend(smiles)
            else:
                all_smiles.append(smiles)
        all_smiles.extend(organic_smiles)

        self.smiles = all_smiles if all_smiles else None
        self.structure = structure

        self.compound_class = assign_class(self.formula)
        try:
            if self.compound_class._type not in ['composite', 'coreshell']:
                self.parts = [self.formula]
            else:
                self.parts = self.compound_class.consist()
        except:
            self.parts = [self.formula]

        self.structure = structure

    def is_inorganic_formula(self, formula):
        """
        Проверяет, является ли формула неорганической
        """
        # Простые критерии для неорганических формул:
        # - содержит только элементы, цифры, скобки, точки
        # - не содержит типичных органических паттернов
        clean_formula = re.sub(r'[-/@]', '', formula)

        # Паттерн для неорганических формул: элементы + цифры + точки
        inorganic_pattern = r'^([A-Z][a-z]?(\d*\.?\,?\d*)*)+$'

        if not re.match(inorganic_pattern, clean_formula):
            return False

        if len(formula) > 30:
            return False

        try:
            elements = get_el_amt_dict(clean_formula)
            return len(elements) > 0
        except:
            return False

    def extract_organic_components(self, formula):
        """
        Разделяет формулу на неорганическую часть и органические компоненты
        Возвращает (inorganic_part, organic_smiles_list)
        """
        if self.is_inorganic_formula(formula):
            return formula, []

        parts = re.split(r'[@]', formula)

        inorganic_parts = []
        organic_smiles_list = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if self.is_inorganic_formula(part):
                inorganic_parts.append(part)
            else:
                smiles = self.get_smiles_from_name(part)
                if smiles:
                    organic_smiles_list.append(smiles)
                else:
                    print(f"Warning: Could not find SMILES for: {part}")

        inorganic_part = '@'.join(inorganic_parts) if inorganic_parts else None
        return inorganic_part, organic_smiles_list

    def get_smiles_from_name(self, name):
        """
        Получает SMILES из названия через pubchempy
        """
        try:
            compounds = pcp.get_compounds(name, namespace='name')
            if compounds:
                return compounds[0].canonical_smiles
        except:
            pass

        try:
                compounds = pcp.get_compounds(name, namespace='synonym')
                if compounds:
                    return compounds[0].canonical_smiles
        except:
                pass

        if name.startswith('PEG-'):
            try:
                compounds = pcp.get_compounds('polyethylene glycol', namespace='name')
                if compounds:
                    return compounds[0].canonical_smiles
            except:
                pass

        return None

    def _apply(self, func):
        results = [func(p) for p in self.parts]
        if len(results) == 1:
            return results[0]
        return sum(results)

    def number_of_atoms(self):
        return sum(get_el_amt_dict(self.formula).values())

    def material_type(self):
        return assign_class(self.formula)._type

    def molecular_weight(self):
        return formula_mass(self.formula)

    def _get_composition(self, formula):
        """Безопасное получение состава с обработкой составных формул"""
        try:
            # Пробуем стандартный pymatgen для простых формул
            return pmg.Composition(formula)
        except:
            # Для составных формул используем наш парсер
            elements = get_el_amt_dict(formula)
            # Создаем Composition из словаря элементов
            return pmg.Composition(elements)

    def average_electronegativity(self):
        def calc_electroneg(formula):
            elements = get_el_amt_dict(formula)
            total_atoms = sum(elements.values())
            if total_atoms == 0:
                return 0
            electroneg_sum = 0
            valid_elements = 0
            for el, amt in elements.items():
                try:
                    # Используем ElementDescriptor для получения электроотрицательности
                    ed = ElementDescriptor(el)
                    electroneg = ed.data.get('Pauling electronegativity')
                    if electroneg is not None:
                        electroneg_sum += electroneg * amt
                        valid_elements += amt
                except:
                    print(f"Warning: Could not get electronegativity for {el}: {e}")
                    continue
            return electroneg_sum / valid_elements if valid_elements > 0 else 0

        return self._apply(calc_electroneg)

    def average_electron_affinity(self):
        elements = get_el_amt_dict(self.formula)
        result = 0
        total_weight = 0
        for el in elements:
            try:
                weight_fraction = get_wt_fraction(self.formula, el)
                ed = ElementDescriptor(el)
                electron_affinity = ed.data.get('Electron affinity_eV')
                if electron_affinity is not None:
                    result += electron_affinity * weight_fraction
                    total_weight += weight_fraction
            except Exception as e:
                print(f"Warning: Could not get electron affinity for {el}: {e}")
                continue
        return result if total_weight > 0 else 0


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
            if hardness != 0:
                softness = 1 / (2 * hardness)
                electrophilicity = (hardness * ((-homo) ** 2)) / (2 * hardness)
            else:
                softness = None
                electrophilicity = None
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
        """
        Рассчитывает атомные и механические дескрипторы
        как средневзвешенные по массе элементарных свойств.
        """
        desc = {}

        total_weight = 0
        for el_symbol, amt in get_el_amt_dict(self.formula).items():
            try:
                #element = pmg.Element(el_symbol)
                ed = ElementDescriptor(el_symbol)
                props = ed.get_numeric()
                weight_frac = get_wt_fraction(self.formula, el_symbol)
                total_weight += weight_frac

                for k, v in props.items():
                    if isinstance(v, (int, float)):
                        desc[k] = desc.get(k, 0.0) + v * weight_frac

            except Exception as e:
                print(f"[WARN] Skipping {el_symbol}: {e}")
                continue

        # Нормализуем, чтобы получились средневзвешенные значения
        if total_weight > 0:
            for k in desc:
                desc[k] /= total_weight

        # Добавим префикс и округлим
        return {f"avg_atomic_{k}": round(v, 6) for k, v in desc.items()}

    def get_thermo_descriptors(self):
        formulas = self.parts

        try:
            weights = [Composition(f).weight for f in formulas]
            total_weight = sum(weights)
            if total_weight != 0:
                weight_fractions = [w / total_weight for w in weights]
        except Exception:
            weights = [1.0 for _ in formulas]
            if len(formulas) !=0:
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
                cat_vals = [d.get(key) for d in descriptors_list if isinstance(d.get(key), str)]
                if cat_vals:
                    averaged["aflow_" + key] = list(set(cat_vals))

        return averaged, descriptors_list

    def E_MSM(self):
        """sum of electrons of the metals and semimetals"""
        e_msm = 0

        for part in self.parts:
            try:
                comp = Composition(part)
                for el in comp.elements:
                    if el.is_metal or el.is_metalloid:
                        e_msm += el.n_electrons * comp[el]
            except Exception as e:
                print(f"Warning in E_MSM for part '{part}': {e}")
                continue

        return {'E_MSM': e_msm}

    def sum_metal_ionization_energy(self):
        """sum of metal ionization energies"""
        smi_en = 0

        for part in self.parts:
            try:
                comp = Composition(part)
                for el in comp.elements:
                    if el.is_metal and el.ionization_energy:
                        smi_en += el.ionization_energy * comp[el]
            except Exception as e:
                print(f"Warning in sum_metal_ionization_energy for part '{part}': {e}")
                continue

        return {'sum_metal_ionization_energy': smi_en}

    def sum_metal_elneg(self):
        """sum of metal electronegativity"""
        metal_elneg_sum = 0

        for part in self.parts:
            try:
                comp = Composition(part)
                for el in comp.elements:
                    if el.is_metal:
                        metal_elneg_sum += el.X * comp[el]
            except Exception as e:
                print(f"Warning in sum_metal_elneg for part '{part}': {e}")
                continue

        if metal_elneg_sum > 0:
            return {'sum_metal_elneg': metal_elneg_sum}
        else:
            return {}

    def sum_metal_elneg_div_ox(self):
        """sum of metal electronegativity divided by oxygen atoms"""
        formula_type = self.compound_class._type

        if formula_type not in ['metal_oxide', 'complex_metal_oxide', 'composite', 'coreshell']:
            return {}

        total_oxygen = 0
        metal_elneg_sum = 0

        for part in self.parts:
            try:
                comp = Composition(part)
                n_ox = comp.get('O', 0)
                total_oxygen += n_ox

                for el in comp.elements:
                    if el.is_metal:
                        metal_elneg_sum += el.X * comp[el]
            except Exception as e:
                print(f"Warning in sum_metal_elneg_div_ox for part '{part}': {e}")
                continue

        if total_oxygen == 0:
            return {}

        smednox = metal_elneg_sum / total_oxygen
        return {'sum_metal_elneg_div_ox': smednox, 'num_oxygen': total_oxygen}


    def all_descriptors(self):
        desc = {
            'number_of_atoms': self.number_of_atoms(),
            'molecular_weight': self.molecular_weight(),
            'average_electronegativity': self.average_electronegativity(),
            'average_electron_affinity': self.average_electron_affinity(),
            'polarizability': self.polarizability(),
            'material_type': self.material_type()
        }
        desc.update(self.homo_lumo())
        desc.update(self.get_thermo_descriptors())
        desc.update(self.smiles_descriptors())
        desc.update(self.electronic_descriptors())
        desc.update(self.structural_descriptors())
        desc.update(self.atomic_mechanical_descriptors())
        desc.update(self.E_MSM())
        desc.update(self.sum_metal_ionization_energy())
        desc.update(self.sum_metal_elneg())
        desc.update(self.sum_metal_elneg_div_ox())
        #aflow_desc, _ = self.aflow_descriptors()
        #desc.update(aflow_desc)
        return desc