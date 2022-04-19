from kamodo import kamodofy, Kamodo
import numpy as np

@kamodofy(units='kg')
def remote_f(x):
    print('remote f called')
    x_ = np.array(x)
    return x_**2 - x_ - 1

kserver = Kamodo(f=remote_f, verbose=True)
server = kserver.serve()

