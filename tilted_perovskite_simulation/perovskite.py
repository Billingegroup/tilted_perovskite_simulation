from diffpy.structure import Structure, Lattice
import numpy as np
import pandas as pd
from tilted_perovskite_simulation.octahedron import Octahedron
from tilted_perovskite_simulation.constants import (
    ATOMIC_NUMBER_TO_SYMBOL,
    ELEMENT_UISO_DICT,
)


class Perovskite:
    def __init__(
        self,
        lattice_param=4.0,  # Default lattice parameter (in Angstroms)
        tilt_angles=None,  # Defaults to [0, 0, 0] if None
        atom_species=None,  # Defaults to ['Ca', 'Ti', 'O'] if None
        uiso=None,
    ):  # Defaults to approximate values if None

        # Check and process lattice_param
        if not isinstance(lattice_param, (int, float)):
            raise ValueError("lattice_param must be an integer or a float.")
        self.lattice_param = float(lattice_param)

        # Process tilt angles: can be list, tuple, or np.array
        if tilt_angles is None:
            self.tilt_angles = np.array([0, 0, 0])
        else:
            if (
                not isinstance(tilt_angles, (list, tuple, np.ndarray))
                or len(tilt_angles) != 3
            ):
                raise ValueError(
                    "tilt_angles must be a list, tuple, or numpy array of three values."
                )
            self.tilt_angles = np.array(tilt_angles)

        # Process atomic species: can be list, tuple, or np.array
        default_atom_species = ["Ca", "Ti", "O"]
        if atom_species is None:
            self.atom_species = np.array(default_atom_species)
        else:
            if (
                not isinstance(atom_species, (list, tuple, np.ndarray))
                or len(atom_species) != 3
            ):
                raise ValueError(
                    "atom_species must be a list, tuple, or numpy array of three values."
                )

            # Convert atomic numbers to element symbols if necessary
            processed_species = []
            for atom in atom_species:
                if isinstance(atom, str):
                    if atom not in ATOMIC_NUMBER_TO_SYMBOL.values():
                        raise ValueError(f"{atom} is not a valid element symbol.")
                    processed_species.append(atom)
                elif isinstance(atom, int):
                    if atom not in ATOMIC_NUMBER_TO_SYMBOL:
                        raise ValueError(f"{atom} is not a valid atomic number.")
                    processed_species.append(ATOMIC_NUMBER_TO_SYMBOL[atom])
                else:
                    raise ValueError(
                        "atom_species entries must be either strings or integers."
                    )
            self.atom_species = np.array(processed_species)

        # Process uiso values: can be list, tuple, or np.array
        if uiso is None:
            # Get default uiso values from the external dict
            self.uiso = np.array(
                [ELEMENT_UISO_DICT[atom] for atom in self.atom_species]
            )
        else:
            if not isinstance(uiso, (list, tuple, np.ndarray)) or len(uiso) != 3:
                raise ValueError(
                    "uiso must be a list, tuple, or numpy array of three values."
                )
            self.uiso = np.array(uiso)

        # Set lattice parameters for cubic 2x2x2 supercell lattice using diffpy Lattice
        self.lattice_supercell = Lattice(
            a=2 * self.lattice_param,
            b=2 * self.lattice_param,
            c=2 * self.lattice_param,
            alpha=90,
            beta=90,
            gamma=90,
        )

        # Generate the 2x2x2 supercell
        self._generate_supercell_structure()
        self.structure_df_cart = self._generate_supercell_dataframe_cart()

    def _generate_supercell_structure(self):
        """
        Generate the supercell structure by creating, rotating, combining the octahedra,
        and updating the lattice parameters.
        """
        self.octahedra = self._initialize_supercell_octahedra()
        self.a_cations = self._initialize_supercell_a_cations()

        # Rotate all octahedra
        for _, octahedron in self.octahedra.items():
            octahedron.rotate(self.tilt_angles)

        # Combine the octahedra to align their corner atoms
        self.octahedra = self._update_octahedra()

        # Update the a cations based on the aligned octahedra
        self.a_cations = self._update_a_cations()

        # Update the lattice parameters based on the aligned octahedra
        self.lattice_supercell = self._update_lattice()

    def _initialize_supercell_octahedra(self):
        """
        Create the 8 octahedra that make up the 2x2x2 supercell and return a dictionary with their positions as keys.
        """
        # Define positions for the 8 octahedra in the supercell
        octahedron_positions = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
        # Additional offset to the center
        offset = np.array([1, 1, 1]) * 0.5 * self.lattice_param
        # Create a dictionary to store octahedra with positions as keys
        octahedra = {}
        for position in octahedron_positions:
            center = np.array(position) * self.lattice_param + offset
            octahedra[position] = Octahedron(
                position=position,
                center=center,
                center_to_corner_length=0.5 * self.lattice_param,
            )

        return octahedra

    def _initialize_supercell_a_cations(self):
        """
        Create the A cations in the 2x2x2 supercell.
        Each A cation is represented as a numpy array of 3 numbers in Cartesian space.
        For any 1 in the position coordinates, it's replaced with 0.5 * self.lattice_param.
        """
        # Define positions for the A cations in the supercell
        cation_positions = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]

        # Create a dictionary to store the A cation positions
        a_cations = {}

        # Initialize each A cation based on the position
        for position in cation_positions:
            # Convert any 1 in the position to 0.5 * self.lattice_param
            cartesian_position = np.array(
                [self.lattice_param if coord == 1 else 0.0 for coord in position]
            )

            # Store the position
            a_cations[position] = cartesian_position

        return a_cations

    def _update_octahedra(self):
        """
        Adjust the centers of the octahedra such that adjacent octahedra overlap at their corner atoms.
        """
        # Calculate the necessary adjustments for each direction using positions as keys
        adjustments = [
            self.octahedra[(1, 0, 0)].atom_positions["left"][0]
            - self.octahedra[(0, 0, 0)].atom_positions["right"][0],  # x-direction
            self.octahedra[(0, 1, 0)].atom_positions["front"][1]
            - self.octahedra[(0, 0, 0)].atom_positions["back"][1],  # y-direction
            self.octahedra[(0, 0, 1)].atom_positions["down"][2]
            - self.octahedra[(0, 0, 0)].atom_positions["up"][2],  # z-direction
        ]

        # Adjust the centers of the octahedra based on their positions
        for position, octahedron in self.octahedra.items():
            # Create an adjustment vector based on the position
            adjustment_vector = np.array(
                [
                    -adjustments[0] if position[0] == 1 else 0,
                    -adjustments[1] if position[1] == 1 else 0,
                    -adjustments[2] if position[2] == 1 else 0,
                ]
            )
            # Apply the adjustment to the center of the octahedron
            octahedron.center += adjustment_vector
            # Apply the same adjustment to each atom position
            for atom_key in octahedron.atom_positions:
                octahedron.atom_positions[atom_key] += adjustment_vector

        return self.octahedra

    def _update_a_cations(self):
        """
        Update the positions of A cations based on the centers of the octahedra.
        For any 1 in the position, the corresponding coordinate is updated based on the difference
        between the centers of adjacent octahedra.
        """

        ref_octa = self.octahedra[(0, 0, 0)]

        for position, cation_pos in self.a_cations.items():
            # Update x-coordinate
            if position[0] == 0:
                cation_pos[0] = ref_octa.atom_positions["left"][0]
            elif position[0] == 1:
                cation_pos[0] = (
                    self.octahedra[(1, 0, 0)].center[0] + ref_octa.center[0]
                ) / 2

            # Update y-coordinate
            if position[1] == 0:
                cation_pos[1] = ref_octa.atom_positions["front"][1]
            elif position[1] == 1:
                cation_pos[1] = (
                    self.octahedra[(0, 1, 0)].center[1] + ref_octa.center[1]
                ) / 2

            # Update z-coordinate
            if position[2] == 0:
                cation_pos[2] = ref_octa.atom_positions["down"][2]
            elif position[2] == 1:
                cation_pos[2] = (
                    self.octahedra[(0, 0, 1)].center[2] + ref_octa.center[2]
                ) / 2

        return self.a_cations

    def _update_lattice(self):
        """
        Update the lattice parameters based on the octahedra.
        """
        # Use the first octahedron (position (0, 0, 0)) as a reference
        reference_octahedron = self.octahedra[(0, 0, 0)]

        # Calculate the new lattice parameters based on the corners of the reference octahedron
        lattice_a = 2 * (
            reference_octahedron.atom_positions["right"][0]
            - reference_octahedron.atom_positions["left"][0]
        )
        lattice_b = 2 * (
            reference_octahedron.atom_positions["front"][1]
            - reference_octahedron.atom_positions["back"][1]
        )
        lattice_c = 2 * (
            reference_octahedron.atom_positions["up"][2]
            - reference_octahedron.atom_positions["down"][2]
        )

        # Update the lattice object with the new parameters
        self.lattice_supercell = Lattice(
            a=lattice_a, b=lattice_b, c=lattice_c, alpha=90, beta=90, gamma=90
        )

        return self.lattice_supercell

    def _generate_supercell_dataframe_cart(self):
        """
        Generate a Pandas DataFrame representing the 2x2x2 supercell structure.
        The DataFrame includes columns: x, y, z in Cartesian coordinates, atom species, and uiso values.

        - First, add all A cations.
        - Then add all centers.
        - Finally, add corner atoms based on position rules.
        """
        data = []

        # 1. Add all A cations (self.atom_species[0])
        data.extend(
            [
                {
                    "x": cation_pos[0],
                    "y": cation_pos[1],
                    "z": cation_pos[2],
                    "atom_species": self.atom_species[0],  # First species for A cations
                    "uiso": self.uiso[0],  # First uiso value
                }
                for cation_pos in self.a_cations.values()
            ]
        )

        # 2. Add all centers (self.atom_species[1])
        data.extend(
            [
                {
                    "x": octahedron.center[0],
                    "y": octahedron.center[1],
                    "z": octahedron.center[2],
                    "atom_species": self.atom_species[1],  # Second species for centers
                    "uiso": self.uiso[1],  # Second uiso value
                }
                for octahedron in self.octahedra.values()
            ]
        )

        # 3. Add corner atoms (self.atom_species[2])
        for position, octahedron in self.octahedra.items():
            corners_to_add = []

            if position[0] == 0:
                corners_to_add.extend(["left", "right"])
            if position[1] == 0:
                corners_to_add.extend(["front", "back"])
            if position[2] == 0:
                corners_to_add.extend(["down", "up"])

            for corner in corners_to_add:
                corner_pos = octahedron.atom_positions[corner]
                data.append(
                    {
                        "x": corner_pos[0],
                        "y": corner_pos[1],
                        "z": corner_pos[2],
                        "atom_species": self.atom_species[
                            2
                        ],  # Third species for corners
                        "uiso": self.uiso[2],  # Third uiso value
                    }
                )

        # Convert the data list to a Pandas DataFrame
        df = pd.DataFrame(data)

        return df

    def get_lattice(self):
        """
        Return the current lattice of the structure.
        """
        return self.lattice_supercell

    def get_structure_dataframe(self, coord_type="cart"):
        """
        Return the structure dataframe. Can output either in Cartesian or fractional coordinates.

        Parameters:
        - coord_type: 'cart' for Cartesian (default) or 'frac' for fractional coordinates.

        Returns:
        - Pandas DataFrame with structure information (x, y, z, atom species, uiso).
        """
        if coord_type == "cart":
            # Return the Cartesian structure DataFrame
            return self.structure_df_cart
        elif coord_type == "frac":
            # Convert Cartesian coordinates to fractional using the inverse of the lattice base
            df = (
                self.structure_df_cart.copy()
            )  # Make a copy of the dataframe to avoid changing the original
            cart_xyz = df[["x", "y", "z"]].values  # Extract Cartesian coordinates
            # Convert to fractional coordinates
            frac_xyz = np.matmul(
                cart_xyz, np.linalg.inv(self.lattice_supercell.stdbase)
            )
            # Update the DataFrame with fractional coordinates
            df[["x", "y", "z"]] = frac_xyz
            return df
        else:
            raise ValueError("coord_type must be either 'cart' or 'frac'.")

    def save_structure_to_cif(self, file_directory):
        """
        Save the structure to a CIF file at the specified file directory.
        """
        # Create the title based on atom species and tilt angles
        title = (
            f"perovskite_{self.atom_species[0]}{self.atom_species[1]}{self.atom_species[2]}3_"
            f"{self.tilt_angles[0]:.2f}_{self.tilt_angles[1]:.2f}_{self.tilt_angles[2]:.2f}"
        )

        # Create a Structure object with the lattice and title
        perovskite_structure = Structure(lattice=self.lattice_supercell, title=title)

        # Get the structure dataframe in fractional coordinates
        structure_df = self.get_structure_dataframe(coord_type="frac")

        # Loop through each row in the dataframe and add atoms to the structure
        for _, row in structure_df.iterrows():
            perovskite_structure.addNewAtom(
                atype=row["atom_species"],
                xyz=row[["x", "y", "z"]].values,  # Fractional coordinates
                occupancy=1.0,
                Uisoequiv=row["uiso"],  # Uiso values
            )

        # Save the structure to the specified CIF file
        perovskite_structure.write(file_directory, format="cif")


"""

        self.build_octahedra()
        self.rotate_octahedra()
        self.fix_displacement()
        self.build_A()
        self.build_structure()

    def get_structure(self):
        return self.structure

    def build_atom_A(self, label: list):
        site = np.array([0.0, 0.0, 0.0])
        if "L" in label:
            site[0] = (
                self.octahedron_FRD.atom_B[0] + self.octahedron_FLD.atom_B[0]
            ) / 2
        if "R" in label:
            site[0] = (
                self.octahedron_FRD.atom_B[0] + self.octahedron_FLD.atom_B[0]
            ) / 2 + 0.5 * self.lattice.a

        if "F" in label:
            site[1] = (
                self.octahedron_BLD.atom_B[1] + self.octahedron_FLD.atom_B[1]
            ) / 2
        if "B" in label:
            site[1] = (
                self.octahedron_BLD.atom_B[1] + self.octahedron_FLD.atom_B[1]
            ) / 2 + 0.5 * self.lattice.b

        if "D" in label:
            site[2] = (
                self.octahedron_FLU.atom_B[2] + self.octahedron_FLD.atom_B[2]
            ) / 2
        if "U" in label:
            site[2] = (
                self.octahedron_FLU.atom_B[2] + self.octahedron_FLD.atom_B[2]
            ) / 2 + 0.5 * self.lattice.c

        return {"site": site, "label": label}



    def fix_displacement(self):
        X = (
            self.octahedron_FRD.get_atom_X()["left"]
            - self.octahedron_FLD.get_atom_X()["right"]
        )
        for octa in self.octahedra:
            if "R" in octa.label:
                octa.translate_atoms(-X)
        self.lattice.a -= 2 * X[0]
        self.lattice.b -= 2 * X[1]
        self.lattice.c -= 2 * X[2]

        Y = (
            self.octahedron_BLD.get_atom_X()["front"]
            - self.octahedron_FLD.get_atom_X()["back"]
        )
        for octa in self.octahedra:
            if "B" in octa.label:
                octa.translate_atoms(-Y)
        self.lattice.a -= 2 * Y[0]
        self.lattice.b -= 2 * Y[1]
        self.lattice.c -= 2 * Y[2]

        Z = (
            self.octahedron_FLU.get_atom_X()["down"]
            - self.octahedron_FLD.get_atom_X()["up"]
        )
        for octa in self.octahedra:
            if "U" in octa.label:
                octa.translate_atoms(-Z)
        self.lattice.a -= 2 * Z[0]
        self.lattice.b -= 2 * Z[1]
        self.lattice.c -= 2 * Z[2]

    def fix_origin_to_B(self, site: np.ndarray):
        return site - self.octahedron_FLD.atom_B

    def build_structure(self):
        perovskite_structure = Structure()
        perovskite_structure.lattice = self.lattice
        lattice_abc = np.array([self.lattice.a, self.lattice.b, self.lattice.c])
        for atom_A in self.A:
            perovskite_structure.addNewAtom(
                atype=self.elements[0],
                xyz=self.fix_origin_to_B(atom_A["site"]) / lattice_abc,
                occupancy=1.0,
                Uisoequiv=self.Uisoequiv[0],
                lattice=self.lattice,
            )
        for octa in self.octahedra:
            perovskite_structure.addNewAtom(
                atype=self.elements[1],
                xyz=self.fix_origin_to_B(octa.get_atom_B()) / lattice_abc,
                occupancy=1.0,
                Uisoequiv=self.Uisoequiv[1],
                lattice=self.lattice,
            )
            X = octa.get_atom_X()
            for atom_X_label in octa.get_atom_X_label_in_cif():
                perovskite_structure.addNewAtom(
                    atype=self.elements[2],
                    xyz=self.fix_origin_to_B(X[atom_X_label]) / lattice_abc,
                    occupancy=1.0,
                    Uisoequiv=self.Uisoequiv[2],
                    lattice=self.lattice,
                )
        self.structure = perovskite_structure

    def save_cif(self, cif_filedir: str):
        self.structure.write(filename=cif_filedir, format="cif")
        return self.glazer_system, self.amplitudes

"""
