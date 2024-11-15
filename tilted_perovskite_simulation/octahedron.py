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
        - processed_tilt_angles: array of adjusted tilt angles in radians. Counterclockwise is positive, clockwise is negative.
        """
        tilt_angles_abs = np.abs(tilt_angles)
        tilt_signs = np.sign(tilt_angles)

        # Define the adjustment factors for each position
        position_adjustments = {
            (0, 0, 0): [1, 1, 1],
            (1, 0, 0): [tilt_signs[0], -1, -1],
            (0, 1, 0): [-1, tilt_signs[1], -1],
            (0, 0, 1): [-1, -1, tilt_signs[2]],
            (1, 1, 0): [-tilt_signs[0], -tilt_signs[1], 1],
            (1, 0, 1): [-tilt_signs[0], 1, -tilt_signs[2]],
            (0, 1, 1): [1, -tilt_signs[1], -tilt_signs[2]],
            (1, 1, 1): [1, 1, 1],
        }

        # Get the adjustment factors based on the current position
        adjustments = position_adjustments.get(self.position, [1, 1, 1])

        # Apply adjustments to the absolute tilt angles
        processed_tilt_angles = tilt_angles_abs * adjustments

        return np.radians(processed_tilt_angles)

    def rotate(self, tilt_angles):
        """
        Apply tilt angles to the octahedron, keeping the center fixed and rotating the X atoms.

        Parameters:
        - tilt_angles: array or list of three tilt angles (in degrees). The first angle represents the tilt
        when looking down the x-axis, specifically the angle of the vector from the center to the right X
        front/back atom relative to the xy-plane. We assume it is equal to the angle of the vector from the
        center to the X up/down atom relative to the xz-plane. The second and third angles apply similarly
        for the y- and z-axes.
        """
        # Process tilt angles based on the octahedron's position
        tilt_angles = self.process_tilt_angles(tilt_angles)

        # Convert tilt angles to radians and separate the absolute values and signs
        tilt_x_rad, tilt_y_rad, tilt_z_rad = np.abs(tilt_angles)
        tilt_x_sign, tilt_y_sign, tilt_z_sign = np.sign(tilt_angles)

        # Rotate the 'right' and 'left' positions
        x = self.center_to_corner_length / np.sqrt(
            1 + np.tan(tilt_y_rad) ** 2 + np.tan(tilt_z_rad) ** 2
        )
        y = x * np.tan(tilt_z_rad)
        z = x * np.tan(tilt_y_rad)
        right_position = np.array([x, -tilt_z_sign * y, tilt_y_sign * z]) + self.center
        left_position = -np.array([x, -tilt_z_sign * y, tilt_y_sign * z]) + self.center
        self.atom_positions["right"], self.atom_positions["left"] = (
            right_position,
            left_position,
        )

        # Rotate the 'back' and 'front' positions
        y = self.center_to_corner_length / np.sqrt(
            1 + np.tan(tilt_x_rad) ** 2 + np.tan(tilt_z_rad) ** 2
        )
        x = y * np.tan(tilt_z_rad)
        z = y * np.tan(tilt_x_rad)
        back_position = np.array([tilt_z_sign * x, y, -tilt_x_sign * z]) + self.center
        front_position = -np.array([tilt_z_sign * x, y, -tilt_x_sign * z]) + self.center
        self.atom_positions["back"], self.atom_positions["front"] = (
            back_position,
            front_position,
        )

        # Rotate the 'up' and 'down' positions
        z = self.center_to_corner_length / np.sqrt(
            1 + np.tan(tilt_x_rad) ** 2 + np.tan(tilt_y_rad) ** 2
        )
        x = z * np.tan(tilt_y_rad)
        y = z * np.tan(tilt_x_rad)
        up_position = np.array([-tilt_y_sign * x, tilt_x_sign * y, z]) + self.center
        down_position = -np.array([-tilt_y_sign * x, tilt_x_sign * y, z]) + self.center
        self.atom_positions["up"], self.atom_positions["down"] = (
            up_position,
            down_position,
        )


"""

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
        rotation_matrix = rot_z @ rot_y @ rot_x

        # Translate atom positions so that the center is at [0, 0, 0], then rotate and translate back
        for key, pos in self.atom_positions.items():
            # Subtract center to bring the octahedron center to the origin
            pos_relative_to_center = pos - self.center
            # Rotate
            rotated_pos = rotation_matrix @ pos_relative_to_center
            # Add the center back after rotation
            self.atom_positions[key] = rotated_pos + self.center

"""
