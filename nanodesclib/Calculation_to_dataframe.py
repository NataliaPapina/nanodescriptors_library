from nanodesclib.AutoML import *
from pathlib import Path
from typing import List, Optional, Union
from pymatgen.core.structure import Structure
import pandas as pd
from nanodesclib.NanoDescriptors import NanoDescriptors


class DescriptorDatasetBuilder:
    def __init__(
        self,
        formulas: Optional[Union[List[str], pd.Series]] = None,
        smiles: Optional[Union[List[Optional[str]], List[List[str]], pd.Series]] = None,
        structures: Optional[Union[List[Optional[Structure]], pd.Series]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        formula_col: str = "formula",
        smiles_cols: Optional[List[str]] = None,
        structure_col: str = "structure"
    ):
        self.original_df = dataframe.copy() if dataframe is not None else None

        if dataframe is not None:
            self.formulas = dataframe[formula_col].tolist()

            if smiles_cols is None:
                smiles_cols = [col for col in dataframe.columns if "smiles" in col.lower()]

            if smiles_cols:
                self.smiles = dataframe[smiles_cols].values.tolist()
            else:
                self.smiles = [[None]] * len(self.formulas)

            self.smiles = [
                [s for s in row if pd.notnull(s)] if isinstance(row, (list, tuple)) else [row]
                for row in self.smiles
            ]

            self.structures = dataframe[structure_col].tolist() if structure_col in dataframe else [None] * len(self.formulas)

        else:
            assert formulas is not None, "Either `formulas` or `dataframe` must be provided."

            self.formulas = list(formulas)

            if smiles is None:
                self.smiles = [[None]] * len(self.formulas)
            elif all(isinstance(s, str) or s is None for s in smiles):
                self.smiles = [[s] for s in smiles]
            else:
                self.smiles = list(smiles)

            self.structures = list(structures) if structures is not None else [None] * len(self.formulas)

            self.original_df = pd.DataFrame({
                "formula": self.formulas,
                "structure": self.structures,
                "smiles": [s[0] if s else None for s in self.smiles]
            })

    def build(self) -> pd.DataFrame:
        all_records = []
        cache = {}

        for idx, (f, smiles_list, struct) in enumerate(zip(self.formulas, self.smiles, self.structures)):
            struct_str = struct.to_pretty_string() if isinstance(struct, Structure) else None
            key = (f, tuple(smiles_list) if smiles_list else None, struct_str)

            if key in cache:
                desc = cache[key]
            else:
                try:
                    desc_obj = NanoDescriptors(formula=f, smiles=smiles_list, structure=struct)
                    desc = desc_obj.all_descriptors()
                    cache[key] = desc
                except Exception as e:
                    print(f"Failed to compute descriptors for {f}: {e}")
                    desc = {}
                    cache[key] = desc
            all_records.append(desc)

        descriptors_df = pd.DataFrame(all_records)
        final_df = pd.concat([self.original_df.reset_index(drop=True), descriptors_df.reset_index(drop=True)], axis=1)
        return final_df

