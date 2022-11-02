def lmap(fn, *x):
    return list(map(fn, *x))

def uniquesorted(x):
    return sorted(set(x))

def isnotnone(x):
    return x is not None

notnone = isnotnone

def zipmap(fn, y):
    return zip(y, map(fn, y))

def dictmap(fn, y):
    return dict(zipmap(fn, y))
