#!/usr/bin/python3

""" parsec.py

Created: 5/7/2015
Author: Michel Robijns

This file is part of parsec which is released under the MIT license.
See the file LICENSE or go to http://opensource.org/licenses/MIT for full
license details.

TODO: Add description
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def main():
    pass


if __name__ == '__main__':
    main()


def PARSEC_to_coefficients(r_LE,
                          x_u, z_u, z_xx_u,
                          x_l, z_l, z_xx_l,
                          alpha_TE, beta_TE,
                          thickness_TE, z_TE):
    """Computes the 12 polynomial coefficients from a set of 11 geometric
       variables
    
    Arguments:
        r_LE: leading edge radius
        x_u: upper crest x-coordinate
        z_u: upper crest z-coordinate
        z_xx_u: upper crest curvature
        x_l: lower crest x-coordinate
        z_l: lower crest z-coordinate
        z_xx_l: lower crest curvature
        alpha_TE: trailing edge direction
        beta_TE: trailing edge wedge angle
        thickness_TE: trailing edge thickness
        z_TE: trailing edge z-coordinate
    
    Returns:
        An array containing the 12 polynomial coefficients
    """
    
    A_u = np.array([
                   [1, 0, 0, 0, 0, 0],
                   [x_u ** 0.5, x_u ** 1.5, x_u ** 2.5, x_u ** 3.5, x_u ** 4.5,
                    x_u ** 5.5],
                   [0.5 * x_u ** -0.5, 1.5 * x_u ** 0.5, 2.5 * x_u ** 1.5,
                    3.5 * x_u ** 2.5, 4.5 * x_u ** 3.5, 5.5 * x_u ** 4.5],
                   [-0.25 * x_u ** -1.5, 0.75 * x_u ** -0.5, 3.75 * x_u ** 0.5,
                    8.75 * x_u ** 1.5, 15.75 * x_u ** 2.5, 24.75 * x_u ** 3.5],
                   [1, 1, 1, 1, 1, 1],
                   [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                   ])
    
    b_u = np.array([math.sqrt(2 * r_LE),
                    z_u,
                    0,
                    z_xx_u,
                    z_TE + 0.5 * thickness_TE,
                    math.tan(alpha_TE - 0.5 * beta_TE)])
                 
    A_l = np.array([
                   [1, 0, 0, 0, 0, 0],
                   [x_l ** 0.5, x_l ** 1.5, x_l ** 2.5, x_l ** 3.5, x_l ** 4.5,
                    x_l ** 5.5],
                   [0.5 * x_l ** -0.5, 1.5 * x_l ** 0.5, 2.5 * x_l ** 1.5,
                    3.5 * x_l ** 2.5, 4.5 * x_l ** 3.5, 5.5 * x_l ** 4.5],
                   [-0.25 * x_l ** -1.5, 0.75 * x_l ** -0.5, 3.75 * x_l ** 0.5,
                    8.75 * x_l ** 1.5, 15.75 * x_l ** 2.5, 24.75 * x_l ** 3.5],
                   [1, 1, 1, 1, 1, 1],
                   [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                   ])
        
    b_l = np.array([-math.sqrt(2 * r_LE),
                    z_l,
                    0,
                    z_xx_l,
                    z_TE - 0.5 * thickness_TE,
                    math.tan(alpha_TE + 0.5 * beta_TE)])
    
    coefficients_u = np.linalg.solve(A_u, b_u)
    coefficients_l = np.linalg.solve(A_l, b_l)
    
    return np.append(coefficients_u, coefficients_l)


def coefficients_to_coordinates(parameters, N=100, half_cosine_spacing=True,
                                save_to_file=False):
    """Generates the coordinates of an airfoil from a set of polynomial coefficients
    
    Arguments:
        coefficients: an array containing the 12 polynomial coefficients
    
    Optional Arguments:
        N: Number of desired airfoil coordinates
        half_cosine_spacing: Half cosine spacing ensures that the datapoints
                             are more widely spaced around the leading edge
                             where the curvature is greatest
        save_to_file: Saves the coordinates in a data file
    
    Returns:
        A matrix with two columns pertaining to the x and y-coordinates,
        respectively. The sequence of coordinates is clockwise, starting at the
        trailing edge.
    """
    
    # Half cosine spacing ensures that the datapoints are more widely spaced
    # around the leading edge where the curvature is greatest.
    if half_cosine_spacing:
        x = (1 - np.cos(np.linspace(0, math.pi, N, dtype=float))) / 2
    else:
        x = np.linspace(0, 1, N)
    
    z_u = z(x, parameters[0:6])
    z_l = z(x, parameters[6:12])
    
    coordinates = np.vstack((np.append(np.flipud(x), x),
                             np.append(np.flipud(z_u), z_l))).T
    
    if save_to_file:
        np.savetxt("PARSEC.dat", coordinates, delimiter='\t', fmt='%f')
    
    return coordinates


def coordinates_to_coefficients(coordinates):
    """Computes the 12 polynomial coefficients from known airfoil coordinates
    
    Arguments:
        coordinates: a matrix containing the airfoil coordinates
        
    Returns:
        coefficients: an array containing the 12 polynomial coefficients
    """
    
    x = coordinates[:, 0]
    z = coordinates[:, 1]

    x_u = x[(np.size(coordinates, 0) / 2):]
    x_l = x_u

    z_u = z[:(np.size(coordinates, 0) / 2 + 1)]
    z_u = np.flipud(z_u)
    z_l = z[(np.size(coordinates, 0) / 2):]

    A_u = np.array([
                   [np.mean(x_u), np.mean(x_u ** 2), np.mean(x_u ** 3),
                    np.mean(x_u ** 4), np.mean(x_u ** 5), np.mean(x_u ** 6),
                    -np.mean(x_l), -np.mean(x_l ** 2), -np.mean(x_l ** 3),
                    -np.mean(x_l ** 4), -np.mean(x_l ** 5),
                    -np.mean(x_l ** 6)],
                   [np.mean(x_u ** 2), np.mean(x_u ** 3), np.mean(x_u ** 4),
                    np.mean(x_u ** 5), np.mean(x_u ** 6), np.mean(x_u ** 7),
                    0, 0, 0, 0, 0, 0],
                   [np.mean(x_u ** 3), np.mean(x_u ** 4), np.mean(x_u ** 5),
                    np.mean(x_u ** 6), np.mean(x_u ** 7), np.mean(x_u ** 8),
                    0, 0, 0, 0, 0, 0],
                   [np.mean(x_u ** 4), np.mean(x_u ** 5), np.mean(x_u ** 6),
                    np.mean(x_u ** 7), np.mean(x_u ** 8), np.mean(x_u ** 9),
                    0, 0, 0, 0, 0, 0],
                   [np.mean(x_u ** 5), np.mean(x_u ** 6), np.mean(x_u ** 7),
                    np.mean(x_u ** 8), np.mean(x_u ** 9), np.mean(x_u ** 10),
                    0, 0, 0, 0, 0, 0],
                   [np.mean(x_u ** 6), np.mean(x_u ** 7), np.mean(x_u ** 8),
                    np.mean(x_u ** 9), np.mean(x_u ** 10), np.mean(x_u ** 11),
                    0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, np.mean(x_l ** 2), np.mean(x_l ** 3),
                    np.mean(x_l ** 4), np.mean(x_l ** 5), np.mean(x_l ** 6),
                    np.mean(x_l ** 7)],
                   [0, 0, 0, 0, 0, 0, np.mean(x_l ** 3), np.mean(x_l ** 4),
                    np.mean(x_l ** 5), np.mean(x_l ** 6), np.mean(x_l ** 7),
                    np.mean(x_l ** 8)],
                   [0, 0, 0, 0, 0, 0, np.mean(x_l ** 4), np.mean(x_l ** 5),
                    np.mean(x_l ** 6), np.mean(x_l ** 7), np.mean(x_l ** 8),
                    np.mean(x_l ** 9)],
                   [0, 0, 0, 0, 0, 0, np.mean(x_l ** 5), np.mean(x_l ** 6),
                    np.mean(x_l ** 7), np.mean(x_l ** 8), np.mean(x_l ** 9),
                    np.mean(x_l ** 10)],
                   [0, 0, 0, 0, 0, 0, np.mean(x_l ** 6), np.mean(x_l ** 7),
                    np.mean(x_l ** 8), np.mean(x_l ** 9), np.mean(x_l ** 10),
                    np.mean(x_l ** 11)],
                   ])

    b_u = np.array([
                   [np.mean(z_u * x_u ** 0.5) - np.mean(z_l * x_l ** 0.5)],
                   [np.mean(z_u * x_u ** 1.5)],
                   [np.mean(z_u * x_u ** 2.5)],
                   [np.mean(z_u * x_u ** 3.5)],
                   [np.mean(z_u * x_u ** 4.5)],
                   [np.mean(z_u * x_u ** 5.5)],
                   [0],
                   [np.mean(z_l * x_l ** 1.5)],
                   [np.mean(z_l * x_l ** 2.5)],
                   [np.mean(z_l * x_l ** 3.5)],
                   [np.mean(z_l * x_l ** 4.5)],
                   [np.mean(z_l * x_l ** 5.5)],
                   ])

    return np.linalg.solve(A_u, b_u)


def coordinates_to_coefficients_experimental(coordinates):
    """Computes the 12 polynomial coefficients from known airfoil coordinates.
       This experimental implementation results in a vastly superior polynomial
       fitting, but the possibility to back-calcualte the 11 PARSEC parameters
       is lost.
    
    Arguments:
        coordinates: a matrix containing the airfoil coordinates
        
    Returns:
        coefficients: an array containing the 12 polynomial coefficients
    """
    
    x = coordinates[:, 0]
    z = coordinates[:, 1]
    
    x_u = x[(np.size(coordinates, 0) / 2):]
    x_l = x_u
    
    z_u = z[:(np.size(coordinates, 0) / 2 + 1)]
    z_u = np.flipud(z_u)
    z_l = z[(np.size(coordinates, 0) / 2):]
    
    A_u = np.array([
                   [np.mean(x_u), np.mean(x_u ** 2), np.mean(x_u ** 3),
                    np.mean(x_u ** 4), np.mean(x_u ** 5), np.mean(x_u ** 6)],
                   [np.mean(x_u ** 2), np.mean(x_u ** 3), np.mean(x_u ** 4),
                    np.mean(x_u ** 5), np.mean(x_u ** 6), np.mean(x_u ** 7)],
                   [np.mean(x_u ** 3), np.mean(x_u ** 4), np.mean(x_u ** 5),
                    np.mean(x_u ** 6), np.mean(x_u ** 7), np.mean(x_u ** 8)],
                   [np.mean(x_u ** 4), np.mean(x_u ** 5), np.mean(x_u ** 6),
                    np.mean(x_u ** 7), np.mean(x_u ** 8), np.mean(x_u ** 9)],
                   [np.mean(x_u ** 5), np.mean(x_u ** 6), np.mean(x_u ** 7),
                    np.mean(x_u ** 8), np.mean(x_u ** 9), np.mean(x_u ** 10)],
                   [np.mean(x_u ** 6), np.mean(x_u ** 7), np.mean(x_u ** 8),
                    np.mean(x_u ** 9), np.mean(x_u ** 10), np.mean(x_u ** 11)],
                   ])
    
    b_u = np.array([
                   [np.mean(z_u * x_u ** 0.5)],
                   [np.mean(z_u * x_u ** 1.5)],
                   [np.mean(z_u * x_u ** 2.5)],
                   [np.mean(z_u * x_u ** 3.5)],
                   [np.mean(z_u * x_u ** 4.5)],
                   [np.mean(z_u * x_u ** 5.5)],
                   ])
    
    A_l = np.array([
                   [np.mean(x_l), np.mean(x_l ** 2), np.mean(x_l ** 3),
                    np.mean(x_l ** 4), np.mean(x_l ** 5), np.mean(x_l ** 6)],
                   [np.mean(x_l ** 2), np.mean(x_l ** 3), np.mean(x_l ** 4),
                    np.mean(x_l ** 5), np.mean(x_l ** 6), np.mean(x_l ** 7)],
                   [np.mean(x_l ** 3), np.mean(x_l ** 4), np.mean(x_l ** 5),
                    np.mean(x_l ** 6), np.mean(x_l ** 7), np.mean(x_l ** 8)],
                   [np.mean(x_l ** 4), np.mean(x_l ** 5), np.mean(x_l ** 6),
                    np.mean(x_l ** 7), np.mean(x_l ** 8), np.mean(x_l ** 9)],
                   [np.mean(x_l ** 5), np.mean(x_l ** 6), np.mean(x_l ** 7),
                    np.mean(x_l ** 8), np.mean(x_l ** 9), np.mean(x_l ** 10)],
                   [np.mean(x_l ** 6), np.mean(x_l ** 7), np.mean(x_l ** 8),
                    np.mean(x_l ** 9), np.mean(x_l ** 10), np.mean(x_l ** 11)],
                   ])
    
    b_l = np.array([
                   [np.mean(z_l * x_l ** 0.5)],
                   [np.mean(z_l * x_l ** 1.5)],
                   [np.mean(z_l * x_l ** 2.5)],
                   [np.mean(z_l * x_l ** 3.5)],
                   [np.mean(z_l * x_l ** 4.5)],
                   [np.mean(z_l * x_l ** 5.5)],
                   ])
    
    coefficients_u = np.linalg.solve(A_u, b_u)
    coefficients_l = np.linalg.solve(A_l, b_l)
    
    return np.append(coefficients_u, coefficients_l)


def z(x, a):
    return (a[0] * x ** 0.5 + a[1] * x ** 1.5 + a[2] * x ** 2.5 +
            a[3] * x ** 3.5 + a[4] * x ** 4.5 + a[5] * x ** 5.5)


def dz(x, a):
    return (0.5 * a[0] * x ** -0.5 + 1.5 * a[1] * x ** 0.5 +
            2.5 * a[2] * x ** 1.5 + 3.5 * a[3] * x ** 2.5 +
            4.5 * a[4] * x ** 3.5 + 5.5 * a[5] * x ** 4.5)


def dz2(x, a):
    return (-0.25 * a[0] * x ** -1.5 + 0.75 * a[1] * x ** -0.5 +
            3.75 * a[2] * x ** 0.5 + 8.75 * a[3] * x ** 1.5 +
            15.75 * a[4] * x ** 2.5 + 24.75 * a[5] * x ** 3.5)


def coefficients_to_PARSEC(coefficients):
    """Computes the 11 PARSEC parameters from a set of 12 polynomial coefficients
    
    Arguments:
        coefficients: n array containing the 12 polynomial coefficients
        
    Returns:
        r_LE: leading edge radius
        x_u: upper crest x-coordinate
        z_u: upper crest z-coordinate
        z_xx_u: upper crest curvature
        x_l: lower crest x-coordinate
        z_l: lower crest z-coordinate
        z_xx_l: lower crest curvature
        alpha_TE: trailing edge direction
        beta_TE: trailing edge wedge angle
        thickness_TE: trailing edge thickness
        z_TE: trailing edge z-coordinate
    """
            
    r_LE = 0.5 * (coefficients[0] ** 2)
    
    x_u = opt.bisect(dz, 0.01, 0.5, args=coefficients[0:6])
    z_u = z(x_u, coefficients[0:6])
    z_xx_u = dz2(x_u, coefficients[0:6])
    
    x_l = opt.bisect(dz, 0.01, 0.5, args=coefficients[6:])
    z_l = z(x_l, coefficients[6:])
    z_xx_l = dz2(x_l, coefficients[6:])
    
    alpha_TE = 0.5 * (math.atan(dz(1, coefficients[0:6])) +
               math.atan(dz(1, coefficients[6:])))
    
    beta_TE = (math.atan(dz(1, coefficients[6:])) -
               math.atan(dz(1, coefficients[0:6])))
    
    z(1, coefficients[0:6])
    
    thickness_TE = z(1, coefficients[0:6]) - z(1, coefficients[6:])
    z_TE = 0.5 * (z(1, coefficients[0:6]) + z(1, coefficients[6:]))
        
    return (r_LE, x_u, z_u, z_xx_u, x_l, z_l, z_xx_l, alpha_TE, beta_TE,
            thickness_TE, z_TE)
