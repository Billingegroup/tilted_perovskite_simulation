# Perovskite Simulation with Octahedral Tilts

This repository simulates perovskite structures with customizable octahedral tilts, atomic species, and other properties. It provides flexibility in input parameters and outputs, including structural information, random sampling, and CIF file export.

generate 2x2x2 supercell
fix 90 degree axis and string length within each octahedron
when rotate each octahedron, we cannot have rotation matrices for each axis, and then do matrix multiplication of them and applied to the octahedron. It will not result in the three amplitudes we hope to achieve and will break the 90 degree axis within each octahedron.
in random sampling provise electron charge
in cif prove **params
## Use Cases

### **Step 1. Creating a Perovskite Structure**

You can create a perovskite structure by specifying several key parameters:

1. **Lattice Parameter**: This represents the side length of the original cubic unit cell. Since the initial perovskite is cubic, only one parameter is required. The lattice parameter must be a positive number, typically in Angstroms (Å). If not provided, a default value `4.0 Å` will be used.

2. **Tilt Angles**: Three numbers representing octahedral tilt angles, in degrees.
   - The order of the tilt angles does not matter, it will give you the same structure, ignoring a change of basis.
    but we will still make the tilt based on your provided order, that is your provided tilted angles will be applied in the order of using a, b, c crystallic lattice axis.
   - **Positive values** indicate an in-phase tilt (rotations in the same direction).
   - **Negative values** indicate an out-of-phase tilt (rotations in opposite directions).
   - If not provided, the default tilt angles are `[0, 0, 0]`.

3. **Atomic Species**: A list of three elements representing atoms A, B, and X in the perovskite structure (ABX₃). This can be provided either as strings (e.g., `['Ca', 'Ti', 'O']`) or as atomic numbers (e.g., `[20, 22, 8]`). Defaults are:
   - Atom A: `'Ca'`
   - Atom B: `'Ti'`
   - Atom X: `'O'`
   The atomic species are specified without considering ionization or charge state.

4. **Atomic Displacement (`uiso`)**: A list of atomic displacement parameters (`uiso`) for each atom type. If not provided, approximate values will be used from predefined constants.

#### Example 1: Default Structure Creation

```python
from tilted_perovskite_simulation.perovskite import Perovskite
# Creating a structure with default values (lattice parameter: 3.905 Å, tilt angles: [0, 0, 0], atomic species: [Ca, Ti, O])
structure = Perovskite()
```

This will create a `CaTiO₃` perovskite structure with default tilt angles (no tilts), lattice parameter, and approximate `uiso` values.

#### Example 2: Custom Tilt Angles, Atomic Species, and Lattice Parameter

```python
tilt_angles = [15, -10, 5]  # In-phase and out-of-phase tilts
atomic_species = ['Ba', 'Zr', 'O']  # Or use atomic numbers: [56, 40, 8]
uiso_values = [0.01, 0.02, 0.03]  # Custom atomic displacement parameters
lattice_param = 4.2  # Custom lattice parameter

# Create a perovskite structure with custom lattice parameter, tilt angles, atomic species, and uiso values
structure = Perovskite(lattice_param=lattice_param, tilt_angles=tilt_angles, atom_species=atomic_species, uiso=uiso_values)
```

#### **Step 2. Structure Output Formats**

The tilted structure in 2x2x2 supercell can be output in the following formats:

1. **Pandas DataFrame**: Columns include x, y, z, atom species, number of electrons, and `uiso` values. Use `structure.structure()` for this format.

2. **Monte Carlo (MC) Random Sampling of Normalized Electron-Weighted Atomic Density Function**:
    You can generate a random sampling of the electron-weighted atomic density, where each atom is treated as a 3D Gaussian.

    ```python
    num_samples = 1000
    density_samples = structure.sample_density(num_samples) # Provides a numpy array with dimension (num_samples, 3)
    ```

3. **CIF File Output**: You can export the structure in CIF format, with the option to add custom entries at the beginning of the CIF file.

   ```python
   file_dir = 'output_structure.cif'
   params = {'author': 'Your Name', 'date': '2024-01-01'}
   structure.write_cif(file_dir, **params)
   ```

4. **Lattice Parameters After Octahedral Tilts**:
   - You can output the lattice parameters after the tilts have been applied. This will return a dictionary containing the lattice constants `a`, `b`, `c` and angles `alpha`, `beta`, `gamma`.

   ```python
   lattice_params = structure.get_lattice_parameters()
   ```
