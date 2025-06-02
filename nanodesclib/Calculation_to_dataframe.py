import pandas as pd
from typing import List, Optional, Union
from pymatgen.core.structure import Structure
from Descriptors import *

class DescriptorDatasetBuilder:
    def __init__(
        self,
        formulas: Optional[List[str]] = None,
        smiles: Optional[Union[List[Optional[str]], List[List[str]]]] = None,
        structures: Optional[List[Optional[Structure]]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        formula_col: str = "formula",
        smiles_cols: Optional[List[str]] = None,
        structure_col: str = "structure"
    ):
        self.extra_columns = None

        if dataframe is not None:
            self.formulas = dataframe[formula_col].tolist()

            if smiles_cols is None:
                smiles_cols = [col for col in dataframe.columns if "smiles" in col.lower()]
            self.smiles = dataframe[smiles_cols].values.tolist() if smiles_cols else [[None]] * len(self.formulas)
            self.smiles = [
                [s for s in row if pd.notnull(s)] if isinstance(row, (list, tuple)) else [row]
                for row in self.smiles
            ]

            self.structures = dataframe[structure_col].tolist() if structure_col in dataframe else [None] * len(self.formulas)

            excluded = {formula_col, structure_col}
            excluded.update(smiles_cols)
            self.extra_columns = dataframe.drop(columns=list(excluded), errors='ignore')
        else:
            assert formulas is not None, "Formulas must be provided"
            self.formulas = formulas

            if smiles is None:
                self.smiles = [[None]] * len(formulas)
            elif all(isinstance(s, str) or s is None for s in smiles):
                self.smiles = [[s] for s in smiles]
            else:
                self.smiles = smiles

            self.structures = structures if structures is not None else [None] * len(formulas)

    def build(self) -> pd.DataFrame:
        all_records = []

        for idx, (f, smiles_list, struct) in enumerate(zip(self.formulas, self.smiles, self.structures)):
            try:
                desc_obj = Descriptors(formula=f, smiles=smiles_list, structure=struct)
                desc = desc_obj.all_descriptors()
                desc["formula"] = f

                if self.extra_columns is not None:
                    extras = self.extra_columns.iloc[idx].to_dict()
                    desc.update(extras)

                all_records.append(desc)
            except Exception as e:
                print(f"Failed to compute descriptors for {f}: {e}")
                continue

        return pd.DataFrame(all_records)