# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the Lesser GNU Public Licence, v2.1 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
:Author:   Joshua L. Adelman, University of Pittsburgh
:Contact:  jla65@pitt.edu

Sample code to use the routine for fast RMSD & rotational matrix calculation.
For the example provided below, the minimum least-squares RMSD for the two
7-atom fragments should be 0.719106 A.

    And the corresponding 3x3 rotation matrix is:

    [[ 0.72216358 -0.52038257 -0.45572112]
     [ 0.69118937  0.51700833  0.50493528]
     [-0.0271479  -0.67963547  0.73304748]]

"""
import numpy as np

import MDAnalysis.lib.qcprot as qcp

from numpy.testing import assert_almost_equal, assert_array_almost_equal
import MDAnalysis.analysis.rms as rms
import pytest


@pytest.fixture()
def atoms_a():
    return np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float64
    )


@pytest.fixture()
def atoms_b():
    return np.array(
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
        dtype=np.float64,
    )


# Calculate rmsd after applying rotation
def rmsd(a, b):
    """Returns RMSD between two coordinate sets a and b."""
    return np.sqrt(np.sum(np.power(a - b, 2)) / a.shape[1])


def test_CalcRMSDRotationalMatrix():
    # Setup coordinates
    frag_a = np.zeros((3, 7), dtype=np.float64)
    frag_b = np.zeros((3, 7), dtype=np.float64)
    N = 7

    frag_a[0][0] = -2.803
    frag_a[1][0] = -15.373
    frag_a[2][0] = 24.556
    frag_a[0][1] = 0.893
    frag_a[1][1] = -16.062
    frag_a[2][1] = 25.147
    frag_a[0][2] = 1.368
    frag_a[1][2] = -12.371
    frag_a[2][2] = 25.885
    frag_a[0][3] = -1.651
    frag_a[1][3] = -12.153
    frag_a[2][3] = 28.177
    frag_a[0][4] = -0.440
    frag_a[1][4] = -15.218
    frag_a[2][4] = 30.068
    frag_a[0][5] = 2.551
    frag_a[1][5] = -13.273
    frag_a[2][5] = 31.372
    frag_a[0][6] = 0.105
    frag_a[1][6] = -11.330
    frag_a[2][6] = 33.567

    frag_b[0][0] = -14.739
    frag_b[1][0] = -18.673
    frag_b[2][0] = 15.040
    frag_b[0][1] = -12.473
    frag_b[1][1] = -15.810
    frag_b[2][1] = 16.074
    frag_b[0][2] = -14.802
    frag_b[1][2] = -13.307
    frag_b[2][2] = 14.408
    frag_b[0][3] = -17.782
    frag_b[1][3] = -14.852
    frag_b[2][3] = 16.171
    frag_b[0][4] = -16.124
    frag_b[1][4] = -14.617
    frag_b[2][4] = 19.584
    frag_b[0][5] = -15.029
    frag_b[1][5] = -11.037
    frag_b[2][5] = 18.902
    frag_b[0][6] = -18.577
    frag_b[1][6] = -10.001
    frag_b[2][6] = 17.996

    # Allocate rotation array
    rot = np.zeros((9,), dtype=np.float64)

    # Calculate center of geometry
    comA = np.sum(frag_a, axis=1) / N
    comB = np.sum(frag_b, axis=1) / N

    # Center each fragment
    frag_a = frag_a - comA.reshape(3, 1)
    frag_b = frag_b - comB.reshape(3, 1)

    # Calculate rmsd and rotation matrix
    qcp_rmsd = qcp.CalcRMSDRotationalMatrix(frag_a.T, frag_b.T, N, rot, None)

    # print 'qcp rmsd = ',rmsd
    # print 'rotation matrix:'
    # print rot.reshape((3,3))

    # rotate frag_b to obtain optimal alignment
    frag_br = np.dot(frag_b.T, rot.reshape((3, 3)))
    aligned_rmsd = rmsd(frag_br.T, frag_a)
    # print 'rmsd after applying rotation: ',rmsd

    assert_almost_equal(
        aligned_rmsd,
        0.719106,
        6,
        "RMSD between fragments A and B does not match excpected value.",
    )

    expected_rot = np.array(
        [
            [0.72216358, -0.52038257, -0.45572112],
            [0.69118937, 0.51700833, 0.50493528],
            [-0.0271479, -0.67963547, 0.73304748],
        ]
    )
    assert_almost_equal(
        rot.reshape((3, 3)),
        expected_rot,
        6,
        "Rotation matrix for aliging B to A does not have expected values.",
    )


def test_innerproduct(atoms_a, atoms_b):
    a = 2450.0
    b = np.array([430, 452, 474, 500, 526, 552, 570, 600, 630])
    number_of_atoms = 4

    e = np.zeros(9, dtype=np.float64)
    g = qcp.InnerProduct(e, atoms_a, atoms_b, number_of_atoms, None)
    assert_almost_equal(a, g)
    assert_array_almost_equal(b, e)


def test_RMSDmatrix(atoms_a, atoms_b):
    number_of_atoms = 4
    rotation = np.zeros(9, dtype=np.float64)
    rmsd = qcp.CalcRMSDRotationalMatrix(
        atoms_a, atoms_b, number_of_atoms, rotation, None
    )  # no weights

    rmsd_ref = 20.73219522556076
    assert_almost_equal(rmsd_ref, rmsd)

    rotation_ref = np.array(
        [
            0.9977195,
            0.02926979,
            0.06082009,
            -0.0310942,
            0.9990878,
            0.02926979,
            -0.05990789,
            -0.0310942,
            0.9977195,
        ]
    )
    assert_array_almost_equal(rotation, rotation_ref, 6)


def test_RMSDmatrix_simple(atoms_a, atoms_b):
    number_of_atoms = 4
    rotation = np.zeros(9, dtype=np.float64)
    rmsd = qcp.CalcRMSDRotationalMatrix(
        atoms_a, atoms_b, number_of_atoms, rotation, None
    )  # no weights

    rmsd_ref = 20.73219522556076
    assert_almost_equal(rmsd_ref, rmsd)

    rotation_ref = np.array(
        [
            0.9977195,
            0.02926979,
            0.06082009,
            -0.0310942,
            0.9990878,
            0.02926979,
            -0.05990789,
            -0.0310942,
            0.9977195,
        ]
    )
    assert_array_almost_equal(rotation, rotation_ref, 6)


def test_rmsd(atoms_a, atoms_b):
    rotation_m = np.array(
        [
            [0.9977195, 0.02926979, 0.06082009],
            [-0.0310942, 0.9990878, 0.02926979],
            [-0.05990789, -0.0310942, 0.9977195],
        ]
    )
    atoms_b_aligned = np.dot(atoms_b, rotation_m)
    rmsd = rms.rmsd(atoms_b_aligned, atoms_a)
    rmsd_ref = 20.73219522556076
    assert_almost_equal(rmsd, rmsd_ref, 6)


def test_weights(atoms_a, atoms_b):
    no_of_atoms = 4
    weights = np.array([1, 2, 3, 4], dtype=np.float64)

    rotation = np.zeros(9, dtype=np.float64)
    rmsd = qcp.CalcRMSDRotationalMatrix(
        atoms_a, atoms_b, no_of_atoms, rotation, weights
    )

    assert_almost_equal(rmsd, 32.798779202159416)
    rotation_ref = np.array(
        [
            0.99861395,
            0.022982,
            0.04735006,
            -0.02409085,
            0.99944556,
            0.022982,
            -0.04679564,
            -0.02409085,
            0.99861395,
        ]
    )
    np.testing.assert_almost_equal(rotation_ref, rotation)
