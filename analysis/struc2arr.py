def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    x = x.view((float, len(x.dtype.names)))
    return x
