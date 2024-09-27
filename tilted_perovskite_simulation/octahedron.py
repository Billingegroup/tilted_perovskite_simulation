from diffpy.structure import Structure, Lattice
import numpy as np


class Octahedron:
    def __init__(self, position, center, center_to_corner_length):
        """
        Initialize the octahedron at a specific position with the given distance from center to corner atom.
        All the atom positions are in Cartisian coordiantes.

        Parameters:
        - position: tuple of 3 coordinates, representing the octahedron's location in the supercell.
        - center_to_corner_length: distance between the center and each corner atom (in Angstroms or other units).
        """
        self.position = position
        self.center = center
        self.center_to_corner_length = center_to_corner_length

        # Initialize the 6 corner atoms with labels for easy tracking
        self.atom_positions = self._generate_atom_positions()

    def _generate_atom_positions(self):
        """
        Generate the initial atomic positions relative to the octahedron center.
        Track 6 atoms based on their relative positions (up, down, left, right, front, back).
        """
        length = self.center_to_corner_length

        # Atom positions relative to the center of the octahedron
        atom_positions = {
            "up": self.center + np.array([0, 0, length]),
            "down": self.center + np.array([0, 0, -length]),
            "left": self.center + np.array([-length, 0, 0]),
            "right": self.center + np.array([length, 0, 0]),
            "front": self.center + np.array([0, -length, 0]),
            "back": self.center + np.array([0, length, 0]),
        }

        return atom_positions

    def process_tilt_angles(self, tilt_angles):
        """
        Process the tilt angles based on the position of the octahedron.
        If a coordinate in the position is odd, the corresponding tilt angle is inverted.

        Parameters:
        - tilt_angles: array or list of three tilt angles (tilt_x, tilt_y, tilt_z).

        Returns:
        - processed_tilt_angles: array of adjusted tilt angles.
        """
        tilt_x, tilt_y, tilt_z = tilt_angles

        # Adjust tilt_x if the first coordinate is odd
        if self.position[0] % 2 != 0:
            tilt_x = -tilt_x

        # Adjust tilt_y if the second coordinate is odd
        if self.position[1] % 2 != 0:
            tilt_y = -tilt_y

        # Adjust tilt_z if the third coordinate is odd
        if self.position[2] % 2 != 0:
            tilt_z = -tilt_z

        return [tilt_x, tilt_y, tilt_z]

    def rotate(self, tilt_angles):
        """
        Apply tilt angles to the octahedron, rotating the atomic positions.

        Parameters:
        - tilt_angles: array or list of three tilt angles (in degrees), corresponding to x, y, z axes.
        """
        # Process tilt angles based on the position of the octahedron
        tilt_angles = self.process_tilt_angles(tilt_angles)

        # Convert tilt angles to radians
        tilt_x_rad, tilt_y_rad, tilt_z_rad = np.radians(tilt_angles)

        # Rotation matrices for each axis (rotation at x-axis for tilt_x, and similarly for y and z)
        rot_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt_x_rad), -np.sin(tilt_x_rad)],
                [0, np.sin(tilt_x_rad), np.cos(tilt_x_rad)],
            ]
        )

        rot_y = np.array(
            [
                [np.cos(tilt_y_rad), 0, np.sin(tilt_y_rad)],
                [0, 1, 0],
                [-np.sin(tilt_y_rad), 0, np.cos(tilt_y_rad)],
            ]
        )

        rot_z = np.array(
            [
                [np.cos(tilt_z_rad), -np.sin(tilt_z_rad), 0],
                [np.sin(tilt_z_rad), np.cos(tilt_z_rad), 0],
                [0, 0, 1],
            ]
        )

        # Apply rotations sequentially (x, y, z order). This means that the rotations are applied first around
        # the x-axis, then the y-axis, and finally the z-axis.
        rotation_matrix = rot_x @ rot_y @ rot_z

        # Translate atom positions so that the center is at [0, 0, 0], then rotate and translate back
        for key, pos in self.atom_positions.items():
            # Subtract center to bring the octahedron center to the origin
            pos_relative_to_center = pos - self.center
            # Rotate
            rotated_pos = rotation_matrix @ pos_relative_to_center
            # Add the center back after rotation
            self.atom_positions[key] = rotated_pos + self.center
