import numpy


def rotate(x, y, a):
    xp = x * numpy.cos(a) - y * numpy.sin(a)
    yp = x * numpy.sin(a) + y * numpy.cos(a)
    return xp, yp
