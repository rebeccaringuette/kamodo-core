import asyncio

import numpy as np

from kamodo import KamodoClient, kamodo

kclient = KamodoClient()

# print(asyncio.run(kclient.f(3)))
print(kclient.f(3))