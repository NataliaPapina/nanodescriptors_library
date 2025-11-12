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
from pymatgen.analysis.local_env import CrystalNN
from nanodesclib.aflow import AflowDescriptors
import numbers
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

current_path = Path(__file__).parent.resolve()
csv_file_path = current_path / "reference/polarizabilities.csv"
polarizabilities = pd.read_csv(csv_file_path)

USE_RDKIT_DESCRIPTORS = False

try:
    if USE_RDKIT_DESCRIPTORS:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem import rdMolDescriptors
        RDKIT_AVAILABLE = True
    else:
        RDKIT_AVAILABLE = False
except ImportError:
    RDKIT_AVAILABLE = False

class NanoDescriptors:
    cas_cache = {}
    thermo_cache = {}

    def __init__(self, formula, smiles=None, structure=None, use_rdkit_descriptors=None):
        self.original_formula = formula

        if not isinstance(formula, str):
            raise ValueError(f"Formula must be a string, got {type(formula)}")

        formula = self.clean_formula(formula)

        # Extract components
        #inorganic_formula, organic_smiles = self.extract_organic_components(formula)

        # Определяем основную формулу для расчетов
        #if inorganic_formula:
         #   self.formula = inorganic_formula
        # else:
        #     self.formula = formula
        #
        # # Combine all SMILES
        # all_smiles = []
        #
        # if smiles:
        #     if isinstance(smiles, list):
        #         all_smiles.extend([s for s in smiles if s])
        #     elif isinstance(smiles, str):
        #         all_smiles.append(smiles)
        #
        # if organic_smiles:
        #     all_smiles.extend([s for s in organic_smiles if s])
        #
        # self.smiles = all_smiles if all_smiles else None
        self.formula = formula
        self.smiles = smiles
        self.structure = structure

        if use_rdkit_descriptors is None:
            self.use_rdkit_descriptors = USE_RDKIT_DESCRIPTORS
        else:
            self.use_rdkit_descriptors = use_rdkit_descriptors

        try:
            self.compound_class = assign_class(self.formula)

            self.parts = self.compound_class.consist()

        except Exception as e:
            print(f"Error in class assignment for {formula}: {e}")
            self.compound_class = Metal(self.formula)
            self.parts = [self.formula]

        if not self.parts:
            self.parts = [self.formula]

    def clean_formula(self, formula):
        """Очищает формулу от невидимых символов и мусора"""
        if not formula:
            return formula

        formula = ''.join(char for char in formula if ord(char) >= 32 and ord(char) <= 126)

        formula = formula.replace('\xa0', '').replace('–', '-').replace('−', '-').strip()

        formula = formula.replace('A0+', '').replace('A0', '')

        formula = formula.strip()

        return formula

    def is_inorganic_formula(self, formula):
        """
        Проверяет, является ли формула неорганической
        """
        # Простые критерии для неорганических формул:
        # - содержит только элементы, цифры, скобки, точки
        # - не содержит типичных органических паттернов
        clean_formula = re.sub(r'[-/@]', '', formula)

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
        """
        if not formula:
            return None, []

        # If it looks like a simple inorganic formula, return as is
        if self.is_inorganic_formula(formula):
            return formula, []

        # For complex formulas with @, try to split
        if '@' in formula:
            parts = formula.split('@')
            inorganic_parts = []
            organic_parts = []

            for part in parts:
                part = part.strip()
                if self.is_inorganic_formula(part):
                    inorganic_parts.append(part)
                else:
                    organic_parts.append(part)

            inorganic_part = '@'.join(inorganic_parts) if inorganic_parts else None

            # Try to get SMILES for organic parts
            organic_smiles = []
            for org_part in organic_parts:
                smiles = self.get_smiles_from_name(org_part)
                if smiles:
                    organic_smiles.append(smiles)
                else:
                    print(f"Warning: Could not find SMILES for: {org_part}")
                    # If we can't find SMILES, treat it as inorganic
                    inorganic_parts.append(org_part)

            inorganic_part = '@'.join(inorganic_parts) if inorganic_parts else None
            return inorganic_part, organic_smiles

        # If no @ and doesn't look inorganic, try to get SMILES
        smiles = self.get_smiles_from_name(formula)
        if smiles:
            return None, [smiles]

        # If all else fails, treat as inorganic
        return formula, []

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
                except Exception as e:
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
            elements = get_el_amt_dict(formula)
            result = 0
            for el in elements:
                pol_row = polarizabilities[polarizabilities['Atom'] == el]
                if not pol_row.empty:
                    try:
                        val = float(pol_row['αD'].values[0].split(' ± ')[0])
                        wt = get_wt_fraction(formula, el)
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

        if total_weight > 0:
            for k in desc:
                desc[k] /= total_weight

        return {f"avg_atomic_{k}": round(v, 6) for k, v in desc.items()}

    def get_thermo_descriptors(self):
        formulas = self.parts

        try:
            weights = [formula_mass(f) for f in formulas]
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
            'molecular_diameter_source', 'LFL_sources', 'LFL_source', 'UFL_sources', 'UFL_source',

            'rings', 'aromatic_rings', 'is_aromatic', 'is_organic', 'is_inorganic', 'is_alkane',
            'is_cycloalkane', 'is_alkene', 'is_alkyne', 'is_aromatic', 'is_alcohol', 'is_amine',
            'is_primary_amine', 'is_secondary_amine', 'is_tertiary_amine', 'is_aldehyde', 'is_ketone',
            'is_carboxylic_acid', 'is_ester', 'is_ether', 'is_nitrile', 'is_nitro', 'is_amide',
            'is_haloalkane', 'is_fluoroalkane', 'is_chloroalkane', 'is_bromoalkane', 'is_iodoalkane',
            'is_imide', 'is_imide', 'is_anhydride', 'is_sulfide', 'is_disulfide', 'is_mercaptan',
            'is_peroxide', 'is_hydroperoxide', 'is_oxime', 'is_isonitrile', 'is_cyanate', 'is_isocyanate',
            'is_thiocyanate', 'is_isothiocyanate', 'is_azide', 'is_azo', 'is_nitroso', 'is_nitrate',
            'is_nitrite', 'is_phosphate', 'is_phosphonic_acid', 'is_phosphodiester', 'is_phosphine',
            'is_sulfonic_acid', 'is_sulfoxide', 'is_sulfone', 'is_sulfonate_ester', 'is_sulfinic_acid',
            'is_thiolester', 'is_thionoester', 'is_thial', 'is_thioketone', 'is_carbothioic_s_acid',
            'is_carbothioic_o_acid', 'is_carbodithioic_acid', 'is_carbamate', 'is_urea', 'is_carbonate',
            'is_orthoester', 'is_orthocarbonate_ester', 'is_siloxane', 'is_boronic_acid', 'is_boronic_ester',
            'is_borinic_acid', 'is_borinic_ester', 'is_acyl_halide', 'is_carboxylic_anhydride',
            'is_alkylaluminium', 'is_alkyllithium', 'is_alkylmagnesium_halide', 'is_silyl_ether',
            'is_phenol', 'is_pyridyl', 'is_polyol', 'is_quat', 'is_branched_alkane', 'is_acid',
            'is_amidine', 'is_imine', 'is_primary_aldimine', 'is_secondary_aldimine', 'is_primary_ketimine',
            'is_secondary_ketimine', 'is_carboxylate', 'is_cyanide', 'is_methylenedioxy',

            # Структурные дескрипторы RDKit
            'Van_der_Waals_volume', 'Van_der_Waals_area', 'similarity_variable'
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

    def get_rdkit_thermo_descriptors(self):
        """Получает только RDKit-зависимые термодинамические дескрипторы"""
        if not self.use_rdkit_descriptors or not RDKIT_AVAILABLE:
            return {}

        formulas = self.parts

        try:
            weights = [formula_mass(f) for f in formulas]
            total_weight = sum(weights)
            if total_weight != 0:
                weight_fractions = [w / total_weight for w in weights]
        except Exception:
            weights = [1.0 for _ in formulas]
            if len(formulas) != 0:
                weight_fractions = [1.0 / len(formulas)] * len(formulas)

        # Только RDKit-зависимые свойства
        rdkit_keys = {
            'rings', 'aromatic_rings', 'is_aromatic', 'is_organic', 'is_inorganic', 'is_alkane',
            'is_cycloalkane', 'is_alkene', 'is_alkyne', 'is_alcohol', 'is_amine',
            'is_primary_amine', 'is_secondary_amine', 'is_tertiary_amine', 'is_aldehyde', 'is_ketone',
            'is_carboxylic_acid', 'is_ester', 'is_ether', 'is_nitrile', 'is_nitro', 'is_amide',
            'is_haloalkane', 'is_fluoroalkane', 'is_chloroalkane', 'is_bromoalkane', 'is_iodoalkane',
            'Van_der_Waals_volume', 'Van_der_Waals_area', 'similarity_variable'
        }

        all_keys = set()
        part_descriptors = []

        for formula, weight_fraction in zip(formulas, weight_fractions):
            try:
                chem = Chemical(formula)
            except Exception:
                continue

            descriptors = {}
            for attr in rdkit_keys:
                try:
                    value = getattr(chem, attr)
                    if callable(value):
                        sig = inspect.signature(value)
                        if len(sig.parameters) > 0:
                            continue
                        value = value()
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
                result['thermo_rdkit_' + key] = np.average(values, weights=weights)

        return result

    def electronic_descriptors(self):
        def calc(formula):
            elements = get_el_amt_dict(formula)
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
                weight = formula_mass(formula)
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
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        atomic_no = ElementDescriptor(el).data.get('Atomic no')
                        if atomic_no is not None:
                            e_msm += atomic_no * amt
            except Exception as e:
                print(f"Warning in E_MSM for part '{part}': {e}")
                continue

        return {'E_MSM': e_msm}

    def sum_metal_ionization_energy(self):
        """sum of metal ionization energies"""
        smi_en = 0

        for part in self.parts:
            try:
                comp = get_el_amt_dict(part)
                for el, amt in comp.items():
                    if el in All_metals:
                        ionization_data = ElementDescriptor(el).data.get('Ionization energies')
                        if ionization_data and len(ionization_data) > 0:
                            first_ionization = ionization_data[0]
                            if first_ionization is not None:
                                smi_en += first_ionization * amt
            except Exception as e:
                print(f"Warning in sum_metal_ionization_energy for part '{part}': {e}")
                continue

        return {'sum_metal_ionization_energy': smi_en}

    def sum_metal_elneg(self):
        """sum of metal electronegativity"""
        metal_elneg_sum = 0

        for part in self.parts:
            try:
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        elneg = ElementDescriptor(el).data.get('Pauling electronegativity')
                        if elneg is not None:
                            metal_elneg_sum += elneg * amt
            except Exception as e:
                print(f"Warning in sum_metal_elneg for part '{part}': {e}")
                continue

        if metal_elneg_sum > 0:
            return {'sum_metal_elneg': metal_elneg_sum}
        else:
            return {}

    def sum_metal_boiling_point(self):
        """sum of metal boiling point"""
        metal_boiling_sum = 0

        for part in self.parts:
            try:
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        boil = ElementDescriptor(el).data.get('Boiling point')
                        if boil is not None:
                            metal_boiling_sum += boil * amt
            except Exception as e:
                print(f"Warning in sum_metal_boiling_point for part '{part}': {e}")
                continue

        if metal_boiling_sum > 0:
            return {'sum_metal_boiling_point': metal_boiling_sum}
        else:
            return {}
    def sum_metal_molecular_weight(self):
        """sum of metal molecular weight"""
        metal_molecular_weight_sum = 0

        for part in self.parts:
            try:
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        weight = formula_mass(el)
                        if weight is not None:
                            metal_molecular_weight_sum += weight * amt
            except Exception as e:
                print(f"Warning in sum_metal_molecular_weight for part '{part}': {e}")
                continue

        if metal_molecular_weight_sum > 0:
            return {'sum_metal_molecular_weight': metal_molecular_weight_sum}
        else:
            return {}

    def part_of_metal_molecular_weight(self):
        """part of metal molecular weight"""
        metal_molecular_weight_sum = 0
        all_metals_list = All_metals.split('|')
        for part in self.parts:
            try:
                el_amt_dict = get_el_amt_dict(part)
                for el, amt in el_amt_dict.items():
                    if el in all_metals_list:
                        weight = formula_mass(el)
                        if weight is not None:
                            metal_molecular_weight_sum += weight * amt
            except Exception as e:
                print(f"Warning in part_of_metal_molecular_weight for part '{part}': {e}")
                continue

        total_mass = formula_mass(self.formula)

        if metal_molecular_weight_sum > 0 and total_mass > 0:
            result = metal_molecular_weight_sum / total_mass
            return {'part_of_metal_molecular_weight': result}
        else:
            return {}

    def average_of_metal_molecular_weight(self):
        """average of metal molecular weight"""
        metal_molecular_weight_sum = 0
        metals = 0
        for part in self.parts:
            try:
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        weight = formula_mass(el)
                        if weight is not None:
                            metal_molecular_weight_sum += weight * amt
                            metals += amt
            except Exception as e:
                print(f"Warning in average_of_metal_molecular_weight for part '{part}': {e}")
                continue

        if metal_molecular_weight_sum > 0:
            return {'average_of_metal_molecular_weight': metal_molecular_weight_sum/metals}
        else:
            return {}

    def average_metal_atomic_radius(self):
        """average of metal atomic radius"""
        metal_atomic_radius_sum = 0
        metals = 0

        for part in self.parts:
            try:
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        rad = ElementDescriptor(el).data.get('Atomic radius')
                        if rad is not None:
                            metal_atomic_radius_sum += rad * amt
                            metals += amt
            except Exception as e:
                print(f"Warning in average_metal_atomic_radius for part '{part}': {e}")
                continue

        if metal_atomic_radius_sum > 0:
            return {'average_metal_atomic_radius': metal_atomic_radius_sum/metals}
        else:
            return {}

    def average_metal_polarizability(self):
        """average of metal polarizability"""
        metal_polarizability_sum = 0
        metals = 0

        for part in self.parts:
            try:
                for el, amt in get_el_amt_dict(part).items():
                    if el in All_metals:
                        result = 0
                        pol_row = polarizabilities[polarizabilities['Atom'] == el]
                        if not pol_row.empty:
                            try:
                                metal_polarizability_sum += float(pol_row['αD'].values[0].split(' ± ')[0])
                                metals += amt
                            except:
                                pass
            except Exception as e:
                print(f"Warning in average_metal_atomic_radius for part '{part}': {e}")
                continue

        if metal_polarizability_sum > 0:
            return {'average_metal_polarizability': metal_polarizability_sum/metals, 'num_metals': metals}
        else:
            return {}

    def sum_metal_elneg_div_ox(self):
        """sum of metal electronegativity divided by oxygen atoms"""
        formula_type = self.compound_class._type

        if formula_type not in ['metal_oxide', 'complex_metal_oxide', 'composite', 'coreshell', 'metallate']:
            return {}

        total_oxygen = 0
        metal_elneg_sum = 0

        for part in self.parts:
            try:
                comp = get_el_amt_dict(part)
                # Проверяем наличие кислорода
                if 'O' in comp:
                    n_ox = comp['O']
                    total_oxygen += n_ox

                for el, amt in comp.items():
                    if el in All_metals:
                        elneg = ElementDescriptor(el).data.get('Pauling electronegativity')
                        if elneg is not None:
                            metal_elneg_sum += elneg * amt
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
        #desc.update(self.smiles_descriptors())
        desc.update(self.electronic_descriptors())
        desc.update(self.structural_descriptors())
        desc.update(self.atomic_mechanical_descriptors())
        desc.update(self.E_MSM())
        desc.update(self.sum_metal_ionization_energy())
        desc.update(self.sum_metal_elneg())
        desc.update(self.sum_metal_elneg_div_ox())
        desc.update(self.sum_metal_boiling_point())
        desc.update(self.average_metal_atomic_radius())
        desc.update(self.average_metal_polarizability())
        desc.update(self.sum_metal_molecular_weight())
        desc.update(self.part_of_metal_molecular_weight())
        desc.update(self.average_of_metal_molecular_weight())
        if self.use_rdkit_descriptors:
            rdkit_desc = self.get_rdkit_thermo_descriptors()
            desc.update(rdkit_desc)
        #aflow_desc, _ = self.aflow_descriptors()
        #desc.update(aflow_desc)
        return desc