import numpy as np
import pytest

from pyofn import OFN
from pyofn.shapes import triangular, triangular_left, trapezoidal, singleton


def test_ofn_subtraction_kosinski():
    # Kosiński's subtraction: A - A = 0
    a = triangular(1.0, 2.0, 3.0, n=64)
    zero_diff = a - a

    support = zero_diff.support
    assert abs(support[0]) < 1e-9
    assert abs(support[1]) < 1e-9
    assert zero_diff.defuzzify_cog() == pytest.approx(0.0, abs=1e-9)


def test_ofn_algebraic_reconstruction():
    # (A + B) - B == A
    a = triangular(1.0, 2.0, 3.0, n=64)
    b = trapezoidal(2.0, 3.0, 4.0, 5.0, n=64)

    reconstructed = (a + b) - b
    assert np.allclose(reconstructed.up, a.up)
    assert np.allclose(reconstructed.down, a.down)


def test_ofn_membership():
    # Membership for trapezoidal(0, 1, 2, 3)
    trap = trapezoidal(0.0, 1.0, 2.0, 3.0, n=64)

    # Core/Plateau check
    assert trap.membership(1.5)[0] == pytest.approx(1.0)
    # Ascending arm check
    assert trap.membership(0.5)[0] == pytest.approx(0.5)
    # Descending arm check
    assert trap.membership(2.5)[0] == pytest.approx(0.5)
    # Outside check
    assert trap.membership(-1.0)[0] == pytest.approx(0.0)
    assert trap.membership(4.0)[0] == pytest.approx(0.0)


def test_ofn_defuzzify_cog():
    # COG of triangular(1, 2, 3) should be 2.0
    tri = triangular(1.0, 2.0, 3.0, n=64)
    assert tri.defuzzify_cog() == pytest.approx(2.0, abs=1e-6)

    # COG of symmetric trapezoid should be the center
    trap = trapezoidal(0.0, 2.0, 4.0, 6.0, n=64)
    assert trap.defuzzify_cog() == pytest.approx(3.0, abs=1e-6)


def test_ofn_direction():
    tri_right = triangular(1.0, 2.0, 3.0, n=64)
    tri_left = triangular_left(1.0, 2.0, 3.0, n=64)
    sing = singleton(5.0, n=64)

    assert tri_right.direction == 1
    assert tri_left.direction == -1
    assert sing.direction == 0


def test_ofn_negation():
    a = triangular(1.0, 2.0, 3.0, n=64)
    neg_a = -a

    # Negation does not swap arms, just negates them pointwise
    assert np.allclose(neg_a.up, -a.up)
    assert np.allclose(neg_a.down, -a.down)
