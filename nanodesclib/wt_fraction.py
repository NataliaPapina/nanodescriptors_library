""" Data _periodic_table.yaml was taken from https://github.com/materialsproject/pymatgen/
                    https://pymatgen.org """

from pathlib import Path
from collections import defaultdict
import re
import yaml

_periodic_table_cache = None

def load_periodic_table():
    global _periodic_table_cache
    if _periodic_table_cache is not None:
        return _periodic_table_cache
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / "reference" / "_periodic_table.yaml"
    with open(file_path, 'r', encoding='utf-8') as f:
        _periodic_table_cache = yaml.safe_load(f)
    return _periodic_table_cache


def element_weight(element):
    data = load_periodic_table()
    mass = data.get('Atomic mass', {}).get('data', {}).get(element)
    if mass is None:
        raise ValueError(f"Atomic mass of {element} wasn't found")
    return float(mass)


def parse_simple_formula(formula):
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    counts = defaultdict(float)
    for el, num in re.findall(pattern, formula):
        counts[el] += float(num) if num else 1.0
    return dict(counts)


def formula_mass(formula):
    counts = parse_simple_formula(formula)
    return sum(element_weight(el) * n for el, n in counts.items())


def get_wt_fraction(formula, element):
    separators = r'[@/–]'
    parts = re.split(separators, formula)

    def parse_part(p):
        m = re.fullmatch(r'\(?([A-Za-z0-9]+)\)?([\d\.]*)', p.strip())
        if m:
            f, factor = m.groups()
            factor = float(factor) if factor else 1.0
            return f, factor
        return p.strip(), 1.0

    total_mass = 0.0
    element_mass = 0.0
    for part in parts:
        formula_part, factor = parse_part(part)
        m_part = formula_mass(formula_part) * factor
        total_mass += m_part
        counts = parse_simple_formula(formula_part)
        if element in counts:
            element_mass += element_weight(element) * counts[element] * factor

    return element_mass / total_mass if total_mass else 0.0
