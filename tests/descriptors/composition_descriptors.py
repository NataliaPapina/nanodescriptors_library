from nanodesclib.NanoDescriptors import *

print(NanoDescriptors('AgCl').formula)
print(NanoDescriptors('AgCl').parts)
print(NanoDescriptors('AgCl').compound_class)
print(NanoDescriptors('AgCl/Pt', smiles='CC').all_descriptors())