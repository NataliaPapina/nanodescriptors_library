from nanodesclib.NanoDescriptors import *

print(NanoDescriptors('AgCl').formula)
print(NanoDescriptors('AgCl').parts)
print(NanoDescriptors('AgCl').compound_class)
print(NanoDescriptors('AgCl/Pt', smiles='CC').all_descriptors())
print(NanoDescriptors('AgCl/Pt').atomic_mechanical_descriptors())
print(NanoDescriptors('AgCl/Pt').average_electron_affinity())
print(NanoDescriptors("Pt0.5Ag0.5").all_descriptors())

test_cases = [
    "Fe@SiO2",
    "Fe-Mn",
    "Fe/MnO2",
    "Fe2O3–SiO2",
    "Fe2O3",
    "invalid-formula"
]

for formula in test_cases:
    try:
        cls = assign_class(formula)
        parts = cls.consist()
        print(f"Formula: {formula}")
        print(f"Class: {cls._type}")
        print(f"Components: {parts}")
        print("-" * 50)
    except Exception as e:
        print(f"Error with {formula}: {e}")
        print("-" * 50)

test_formulas = ["Fe3O4", "MnO2", "CuO", "AgCl/Pt", "Fe-Mn", "WO2.92", "Pt0.5Ag0.5", "Au0.75Pt0.25"]

for formula in test_formulas:
    try:
        nd = NanoDescriptors(formula)
        print(f"Formula: {formula}")
        print(f"Class: {nd.compound_class._type}")
        print(f"Components: {nd.parts}")
        print(f"Number of atoms: {nd.number_of_atoms()}")
        print(f"Molecular weight: {nd.molecular_weight()}")
        print("-" * 50)
    except Exception as e:
        print(f"Error with {formula}: {e}")
        print("-" * 50)

problematic_formulas = [
    "Pt3-Ru", "Pt-Ru", "Pt-Ru3",
    "Ag0.26-Au0.3", "Co3O4-CeO2",
    "Fe3O4", "MnO2", "AgCl/Pt"
]

for formula in problematic_formulas:
    try:
        parts = normalize_formula(formula)
        cls = assign_class(formula)
        print(f"Formula: {formula}")
        print(f"Normalized components: {parts}")
        print(f"Class: {cls._type}")
        print("-" * 60)
    except Exception as e:
        print(f"Error with {formula}: {e}")
        print("-" * 60)

# Тестируем
test_formulas = [
    "Fe1Fe2O4@PEG-550",
    "Fe1Fe2O4@PEG-2000",
    "Dy1Si1O2@sulfo-SMCC",
    "Fe1Pt1@DSPE-PEG5000-FA",
    "Fe1Fe2O4@NDOPA-PEG",
    "SiO2",  # обычная неорганическая
    "C6H12O6",  # органическая (глюкоза)
]

for formula in test_formulas:
    inorganic, organic_smiles = NanoDescriptors(formula).extract_organic_components(formula)
    print(f"Formula: {formula}")
    print(f"  Inorganic: {inorganic}")
    print(f"  Organic SMILES: {organic_smiles}")
    print()