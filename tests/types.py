from nanodesclib.assign_class import *


assert assign_class('Au2.0O@PdCl2')._type == 'coreshell'
assert assign_class('Au2.0O-PdCl2')._type == 'composite'
assert assign_class('Au2.0O/PdCl2')._type == 'composite'
assert assign_class('Au2.0O')._type == 'metal_oxide'
assert assign_class('PdCl2')._type == 'salt'
assert assign_class('CuO')._type == 'metal_oxide'
assert assign_class('C')._type == 'non_metal'
assert assign_class('Fe2C3')._type == 'carbide'
assert assign_class('Pd@Pt')._type == 'coreshell'
assert assign_class('Ce2(CO3)2O')._type == 'salt'
assert assign_class('Fe4(Fe(CN)6)3')._type == 'complex_salt'
assert assign_class('Fe4AgPt0.2W0.01')._type == 'tetrametal'

print('Class Au2.0O@PdCl2:', CoreShell('Au2.0O@PdCl2'))
print('Au2.0O@PdCl2 consists of:', CoreShell('Au2.0O@PdCl2').consist())
print(CoreShell('Au2.0O@PdCl2').formula)
print(CoreShell('Au2.0O@PdCl2')._type)
print(assign_class('Mn0.34Fe0.66Fe2O4'))