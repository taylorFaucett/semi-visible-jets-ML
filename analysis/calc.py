import numpy


def mass_inv(j1, j2):
    return numpy.sqrt(
        2.0 * j1.pt * j2.pt * (numpy.cosh(j1.eta - j2.eta) - numpy.cos(j1.phi - j2.phi))
    )
