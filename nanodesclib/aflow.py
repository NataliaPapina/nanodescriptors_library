import requests
import json
import pandas as pd
from pymatgen.core import Composition

class AflowDescriptors:
    def __init__(self, formula: str):
        self.formula = formula
        self.species = list(Composition(formula).as_dict().keys())
        self.descriptors = {}

    def fetch(self, page_size: int = 1000):
        props = [
            "compound", "spacegroup_relax", "spacegroup_orig",
            "volume_atom", "volume_cell", "density",
            "energy_atom", "energy_cell", "Egap", "Egap_type"
        ]
        species_str = ",".join(self.species)
        props_str = ",".join(props)
        url = f"https://aflowlib.duke.edu/search/API/?species({species_str}),paging(0,{page_size}),{props_str}"
        print(f"Fetching from:\n{url}\n")

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return

        try:
            raw_json = json.loads(response.text)
            entries = list(raw_json.values()) if isinstance(raw_json, dict) else raw_json
            df = pd.DataFrame(entries)
        except Exception as e:
            print(f"Error: {e}")
            return

        if df.empty or 'compound' not in df.columns:
            print("No matches.")
            return

        df = df[df['compound'].apply(self._is_equivalent)]
        if df.empty:
            print("No structures", self.formula)
            return

        df = df.sort_values("energy_atom").reset_index(drop=True)
        best = df.iloc[0]

        for col in props:
            self.descriptors[col] = best.get(col, None)

    def _is_equivalent(self, compound: str) -> bool:
        try:
            return Composition(compound).reduced_formula == Composition(self.formula).reduced_formula
        except:
            return False

    def get_descriptors(self) -> dict:
        return self.descriptors
