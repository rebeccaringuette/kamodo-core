# RPC

Kamodo includes a Remote Call Procedure (RPC) interface, based on [capnproto](https://capnproto.org/). This allows Kamodo objects to both serve and connect to other kamodo functions hosted on external servers.

```python
from kamodo import Kamodo, kamodofy
import numpy as np
```

First we'll initialize a Kamodo object to act as a server.

```python
kserver = Kamodo(f='sqrt((x**2-x-1)**2)',
                 g = kamodofy(lambda x=np.linspace(-1, 2, 75): np.sin(x)),
                 verbose=True)
kserver.to_latex()
```

\\begin{equation}f{\\left(x \\right)} = \\sqrt{\\left(x^{2} - x - 1\\right)^{2}}\\end{equation} \\begin{equation}g{\\left(x \\right)} = \\lambda{\\left(x \\right)}\\end{equation}

```python
kserver.detail()
```

```python
kserver.f(np.array([3, 4, 5]))

# >>> array([ 5., 11., 19.])
```

```python
import numpy as np
kserver.plot('g', f=dict(x=np.linspace(-1,2,33)))
```

We can test the RPC features using a read/write socket pair.

```python
import socket

read, write = socket.socketpair()
```

```python
server = kserver.server(write)
```

```python
kserver.signatures
```

Now we'll initialize an empty kamodo object to act as a client.

```python
kclient = Kamodo()

client = kclient.client(read)

kclient.to_latex()
```

\\begin{equation}f{\\left(x \\right)} = \\sqrt{\\left(x^{2} - x - 1\\right)^{2}}\\end{equation} \\begin{equation}g{\\left(x \\right)} = <function <lambda> at 0x7feb79c98710>\\end{equation}

```python
kclient.g
```

The client now has access to the same function hosted on the server. This allows the client to call the server-side function without needing any of its dependencies!

```python
kclient.g().shape # client function inherits defaults of server-side function

# >>> (75,)
```

```python
kclient.g
```

```python
kserver.g
```

Arguments are automatically converted to numpy arrays before being passed to the server.

```python
kclient.f([3, 4, 5])

# >>> array([ 5., 11., 19.])
```

We can plot the function over our custom domain.

```python
kserver.g.meta
```

```python
kserver.to_latex('g')
```

```python
kserver.detail()
```

```python
kclient.plot('g', f=dict(x=np.linspace(-1, 2, 33)))
```

<!-- #region -->
# RPC Spec

Kamodo's rpc specification file is located in `kamodo/rpc/kamodo.capnp`:


```sh
{! ../kamodo/rpc/kamodo.capnp !}
```
<!-- #endregion -->
