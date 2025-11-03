from pathlib import Path
from collections import defaultdict
import re
import yaml

_periodic_table_cache = None

def load_periodic_table():
    global _periodic_table_cache
    if _periodic_table_cache is not None:
        return _periodic_table_cache
    file_path = Path(__file__).parent / "reference" / "_periodic_table.yaml"
    with open(file_path, 'r', encoding='utf-8') as f:
        _periodic_table_cache = yaml.safe_load(f)
    return _periodic_table_cache

def element_weight(element):
    """Возвращает атомную массу элемента"""
    data = load_periodic_table()
    mass = data.get('Atomic mass', {}).get('data', {}).get(element)
    if mass is None:
        raise ValueError(f"Атомная масса для {element} не найдена")
    return float(mass)

def parse_formula(formula):
    """
    Разбирает простую формулу H2O, Fe2O3, C6H12O6
    """
    pattern = r'([A-Z][a-z]?)(\d*\.?\,?\d*)'
    counts = defaultdict(float)
    for el, num in re.findall(pattern, formula):
        counts[el] += float(num) if num else 1.0
    return dict(counts)

def split_components(formula_str):
    """
    Разбивает смесь на компоненты с коэффициентами
    """
    s = re.sub(r'[-/@]', '|', formula_str)
    pattern = r'\(([^()]+)\)([0-9.]+)|([A-Z][a-z]?[A-Za-z0-9]*)'
    parts = []
    for match in re.finditer(pattern, s):
        if match.group(1):
            formula = match.group(1)
            factor = float(match.group(2))
        else:
            formula = match.group(3)
            factor = 1.0
        parts.append((formula, factor))
    return parts

def formula_mass(formula):
    counts = parse_formula(formula)
    return sum(element_weight(el) * n for el, n in counts.items())

def get_wt_fraction(formula_str, element):
    """
    Упрощенная версия для массовой доли элемента
    """
    comps = split_components(formula_str)

    total_mass = 0
    elem_mass = 0

    for f, fct in comps:
        counts = parse_formula(f)
        comp_mass = formula_mass(f) * fct
        total_mass += comp_mass

        if element in counts:
            elem_mass += element_weight(element) * counts[element] * fct

    return elem_mass / total_mass if total_mass > 0 else 0.0

def get_component_fractions(formula_str):
    """
    Для долей компонентов (используется в классификации)
    """
    comps = split_components(formula_str)
    total_mass = sum(formula_mass(f) * fct for f, fct in comps)
    return {f"{f}": formula_mass(f) * fct / total_mass for f, fct in comps}
