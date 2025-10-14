""" Data _periodic_table.yaml was taken from https://github.com/materialsproject/pymatgen/
                    https://pymatgen.org """

import ast, json, yaml
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def load_periodic_table():
    file_path = Path(__file__).resolve().parent / "reference" / "_periodic_table.yaml"
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_element_property(data, element, property_name):
    if property_name in data:
        prop_data = data[property_name]
        if 'data' in prop_data and element in prop_data['data']:
            return prop_data['data'][element]
    return None


def get_element_property_with_unit(data, element, property_name):
    value = get_element_property(data, element, property_name)
    if value is not None and property_name in data:
        prop_data = data[property_name]
        unit = prop_data.get('unit', '')
        return value, unit
    return value, ''


def list_available_properties(data):
    return list(data.keys())


def try_parse_string(value_str):
    if not isinstance(value_str, str):
        return value_str
    value_str = value_str.strip()
    if not value_str:
        return None
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(value_str)
        except Exception:
            pass
    try:
        return float(value_str)
    except ValueError:
        pass
    try:
        return int(value_str)
    except ValueError:
        pass
    return value_str


def element_descriptors(element):
    data = load_periodic_table()
    desc = {}
    for prop in list_available_properties(data):
        value, unit = get_element_property_with_unit(data, element, prop)
        if value is not None:
            key = f"{prop}_{unit}" if unit else prop
            parsed = try_parse_string(value)
            desc[key] = parsed
    return desc


class ElementDescriptor:
    def __init__(self, element):
        self.element = element
        self.data = element_descriptors(element)

    def get_numeric(self):
        numeric = {}
        for k, v in self.data.items():
            if isinstance(v, (int, float)):
                numeric[k] = v
        return numeric

    def get_orbital_energy(self, orbital):
        return self.data.get("Atomic orbitals_hartree", {}).get(orbital)

    def get_ionic_radius(self, charge):
        return self.data.get("Ionic radii_ang", {}).get(str(charge))


