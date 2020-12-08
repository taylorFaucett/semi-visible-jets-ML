import numpy


def heaviside(x):
    return 0.5 * (numpy.sign(x) + 1)


def ado_calc(fx0, fx1, gx0, gx1):
    dos = do_calc(fx0, fx1, gx0, gx1)
    ado_val = numpy.mean(dos)
    if ado_val < 0.5:
        ado_val = 1.0 - ado_val
    return ado_val


def do_calc(fx0, fx1, gx0, gx1):
    dfx = fx0 - fx1
    dgx = gx0 - gx1
    dos = heaviside(numpy.multiply(dfx, dgx))
    return dos
