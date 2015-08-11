#!/usr/bin/python3

""" main.py

Created: 5/7/2015
Author: Michel Robijns

This file is part of parsec which is released under the MIT license.
See the file LICENSE or go to http://opensource.org/licenses/MIT for full
license details.

TODO: Add description
"""

import parsec
from naca import NACA4
import matplotlib.pyplot as plt


def main():
    # Generate some airfoil coordinates
    coordinates = NACA4('2412', 50)
    
    # Convert the airfoil coordinates to the 12 polynomial coefficients
    coefficients = parsec.coordinates_to_coefficients(coordinates)
    
    # Convert the 12 polynomial coefficients to 11 PARSEC parameters
    (r_LE, x_u, z_u, z_xx_u, x_l, z_l, z_xx_l, alpha_TE, beta_TE,
     thickness_TE, z_TE) = parsec.coefficients_to_PARSEC(coefficients)
    
    # Convert the 11 PARSEC parameters to 12 polynomial coefficients
    coefficients_new = parsec.PARSEC_to_coefficients(r_LE,
                                                     x_u, z_u, z_xx_u,
                                                     x_l, z_l, z_xx_l,
                                                     alpha_TE, beta_TE,
                                                     thickness_TE, z_TE)
    
    # Convert the 12 polynomial coefficients to airfoil coordinates
    coordinates_new = parsec.coefficients_to_coordinates(coefficients_new)
    
    # Plot the original and the final coordinates to verify that the process
    # works. Note that the final coordinates do not match exactly because some
    # information is lost when the 12 polynomial coefficients are converted to
    # the 11 PARSEC parameters.
    plt.plot(coordinates[:, 0], coordinates[:, 1], color='red')
    plt.plot(coordinates_new[:, 0], coordinates_new[:, 1], color='blue')
    plt.show()
    
    # As an alternative, you could use the 12 polynomial coefficients
    # instead of the 11 PARSEC parameters. This completely does away with
    # the purpose of why PARSEC was conceived, but it DOES result in
    # a very nice polynomial fitting. Here's an example:
    
    # Generate some airfoil coordinates
    coordinates = NACA4('2412', 50)
    
    # Convert the airfoil coordinates to the 12 polynomial coefficients
    coefficients = parsec.coordinates_to_coefficients_experimental(coordinates)
    
    # Convert the 12 polynomial coefficients to airfoil coordinates
    coordinates_new = parsec.coefficients_to_coordinates(coefficients)
    
    # Plot the original and the final coordinates
    plt.plot(coordinates[:, 0], coordinates[:, 1], color='red')
    plt.plot(coordinates_new[:, 0], coordinates_new[:, 1], color='blue')
    plt.show()


if __name__ == '__main__':
    main()
