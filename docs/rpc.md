# RPC

Kamodo includes a Remote Call Procedure (RPC) interface, based on [capnproto](https://capnproto.org/). This allows Kamodo objects to both serve and connect to other kamodo functions hosted on external servers.

```python
from kamodo import Kamodo, kamodofy
import numpy as np
```

```python
k = Kamodo('f=x**2+y**2')
```

```python
k.evaluate('f', x=np.array([3,4,5]), y=np.array([6,7,8]))
```

First we'll initialize a Kamodo object to act as a server, using one pure function `f` initialized without defaults and one black-box function `g` with defaults.

```python
kserver = Kamodo(f='sqrt((x**2-x-1)**2)',
                 g = kamodofy(lambda x=np.linspace(-1, 2, 12): np.sin(np.array(x))),
                cat = 'a+b')
kserver.to_latex()
kserver
```

\\begin{equation}f{\\left(x \\right)} = \\sqrt{\\left(x^{2} - x - 1\\right)^{2}}\\end{equation} \\begin{equation}g{\\left(x \\right)} = \\lambda{\\left(x \\right)}\\end{equation}


As a test, we'll evaluate `f` on the server.

```python
kserver.f(np.array([3, 4, 5]))

# >>> array([ 5., 11., 19.])
```

Now generate a server-side test plot for both functions.

```python
import plotly.io as pio
fig = kserver.plot('g', f=dict(x=np.linspace(-1,2,33)))
pio.write_image(fig, 'notebooks/images/rpc-plot.svg')
fig
```

![rpc_plot1](notebooks/images/rpc-plot.svg)


To test the RPC features, we'll create a read/write socket pair.

```python
import socket

read, write = socket.socketpair()
```

```python
server = kserver.server(write)
```

Now we'll initialize an empty kamodo object to act as a client.

```python
kclient = Kamodo()

client = kclient.client(read)

kclient.to_latex()
```

\\begin{equation}f{\\left(x \\right)} = \\sqrt{\\left(x^{2} - x - 1\\right)^{2}}\\end{equation} \\begin{equation}g{\\left(x \\right)} = \\lambda{\\left(x \\right)}\\end{equation} \\begin{equation}\\operatorname{cat}{\\left(a,b \\right)} = a + b\\end{equation}


The client now has access to the functions hosted on the server. This allows the client to call server-side functions without needing any of their dependencies!


Client functions inherit the defaults of their server-side counterparts.

```python
from kamodo import get_defaults

get_defaults(kclient.g)

# >>> {'x': array([-1.        , -0.72727273, -0.45454545, -0.18181818,  0.09090909,
#          0.36363636,  0.63636364,  0.90909091,  1.18181818,  1.45454545,
#          1.72727273,  2.        ])}
```

Typecasting will only be handled by the server-side function implementation.

```python
kclient.f(np.array([3, 4, 5]))

# >>> array([ 5., 11., 19.]) # matches above evaluation on server
```

```python
from capnp import KjException
try:
    kclient.f([3, 4, 5])
except KjException as m: 
    #raises remote exception TypeError
    pass
```

```python
# g implementation happens to cast argument to np.array
assert (kclient.g([3, 4, 5]) == kserver.g([3, 4, 5])).all()
```

The client can send other types besides arrays, provided the server-side implementation supports them.

```python
kclient.cat('two', 'three')
```

```python
kclient.cat(['two'], ['three'])
```

We can verify that the clientside plots reproduce that found on the server.

```python
fig = kclient.plot('g', f=dict(x=np.linspace(-1,2,33)))

pio.write_image(fig, 'notebooks/images/rpc-plot2.svg')
fig
```

![rpc_plot1](notebooks/images/rpc-plot2.svg)


# Pipelining

To fascilitate pipelining, the server needs to wrap all function results in promises


# Collecting terms

A client can point to terms located on different servers. Suppose the client wants to evaluate the following expression:

$$ f(x,y,z) = g(x) + h_1(y) + h_2(z)$$

where $h_1$ and $h_2$ are on the same server. In order to minimize data transfers, we want to execute $h_1(y) + h_2(z)$ on the server, retrieve the result and add it to $g(x)$. I don't know how to automatically do this. However, one approach would be to split $f$ into two functions $f_1$ and $f_2$:

$$ f_2(y,z) = h_1(y) + h_2(z) $$
$$ f_1(x,y,z) = g(x) + f_2(y,z) $$

Now we could detect if all terms are on the same server. If so, we can pipeline the whole expression.

A KamodoClient would do all of the above by default. It would only point to one server. Since a kamodo object can be composed of functions coming from other Kamodo objects, this provides a natural separation of resposibilities. It also offloads handling name collisions to the end user.


### Function

```python
from kamodo.rpc.proto import FunctionRPC
```

```python
f = FunctionRPC(lambda x=np.linspace(-3, 3, 33): x**2)
f.getArgs()
```

```python
from kamodo.rpc.proto import Value
import numpy as np
```

### Value

```python
from kamodo.rpc.proto import Value, to_rpc_literal, from_rpc_literal
```

```python
v = Value(np.linspace(-5,5,12))
```

```python
v.read().to_dict()
```

### Parameter

```python
param = to_rpc_literal(np.linspace(-5,5,12))
param.to_dict()
```

### Expression


RPC expressions can refer to functions or values.

```python
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('/Users/asherp/git/ensemblegovservices/kamodo-core/kamodo/rpc/kamodo.capnp')
Expression = kamodo_capnp.Kamodo.Expression
```

```python
literal = Expression(literal=param)
```

```python
expr = Expression(call=dict(function=f, params=[]))
```

```python
expr.which()
```

```python
expr.to_dict()
```

### Evaluate

Regardless of expression type, `evaluate` returns a `Value`.

```python
from kamodo.rpc.proto import KamodoRPC
```

```python
k = KamodoRPC(verbose=True)
```

```python
expr = Expression(call=dict(function=f, params=[literal]))
expr.to_dict()
```

```python
eval_promise = kclient._client.evaluate(expr)

read_promise = eval_promise.value.read().wait()

from_rpc_literal(read_promise.value)
```

```python
eval_promise = kclient._client.evaluate(Expression(literal=param))

read_promise = eval_promise.value.read().wait()

from_rpc_literal(read_promise.value)
```

### Sympy to RPC expression

The client will generate expressions that will be turned into Expression messages to be sent and executed on the server.


```python
from kamodo.rpc.proto import to_rpc_expr
help(to_rpc_expr)
```

```python
from sympy import sympify
```

```python
expr = sympify('x**2-x-1')
expr
```

```python
from kamodo.rpc.proto import add_rpc, mul_rpc, pow_rpc
from sympy import Add, Mul, Pow
```

```python
math_rpc = {Add: add_rpc, Mul:mul_rpc, Pow: pow_rpc}
```

```python
from sympy.abc import a,b,c
expr = a**3+b*c*4.2
to_rpc_expr(expr, a=4, b=5, c=3).to_dict()
```

```python
try:
    to_rpc_expr(expr, a=4, b=5)
except TypeError as m:
    print(m)
```

Evaluate the expression

```python
expr = sympify('x + y + g(x)')
expr
```

```python
def myfunc(x):
    print('function called with param {}'.format(x))
    return x**2
```

```python
literal.to_dict()
```

```python
from kamodo.rpc.proto import math_rpc, Float, Integer, Symbol

rpc_expr = to_rpc_expr(sympify('x+a/b+f(a,x)'),
                       a=3,
                       b=2,
                       x=np.linspace(-5,5,12),
                       f=FunctionRPC(lambda a, x: a+x))

eval_promise = kclient._client.evaluate(rpc_expr)

read_promise = eval_promise.value.read()

from_rpc_literal(read_promise.wait().value)
```

Custom functions can be included on client or server. See calculator example.

```python
from kamodo.rpc.proto import FunctionRPC
```

```python
expr = sympify('a+b + g(x)')
expr
```

## KamodoClient
Our previous example demonstrated running a remote kamodo function from the client. However, compositions involving those functions will be carried out locally:

```python
kclient['h'] = 'g+f'
kclient
```

```python
assert kclient.h(3) == np.sin(np.array(3)) + ((3**2-3-1)**2)**.5
```

Here, the results for `f(x)` and `g(x)` are returned to the client and the client adds the two results. This increases the amount of data being sent. Instead, we can have the server execute the composition requested by the client, so only the final calculation is returned.

The KamodoClient is a subclass of Kamodo that can send and receive server-side compositions.
The KamodoClient will include FunctionRPC for local functions. This allows the server to call them where necessary.

```python
from kamodo.rpc.proto import rpc_map_to_dict
```

```python
from sympy.abc import a,b,c

def get_remote_composition(self, expr, **kwargs):
    """Generate a callable function composition that is executed remotely"""
    def remote_composition(**params):
        remote_expr = to_rpc_expr(expr, **params, **kwargs)
        evaluated = self._client.evaluate(remote_expr).wait()
        result_message = evaluated.value.read().wait()
        result = from_rpc_literal(result_message.value)
        return result
    return remote_composition

myfunc = get_remote_composition(kclient, a+b*c)
myfunc(a=3, b=np.array([3, 4, 2]), c=2.)
```

```python
from kamodo.util import sign_defaults
import capnp

from kamodo.rpc.proto import kamodo_capnp, rpc_map_to_dict, to_rpc_literal, from_rpc_literal

import forge

from kamodo.util import construct_signature, get_undefined_funcs

from kamodo.rpc.proto import FunctionRPC

from kamodo import Kamodo, kamodofy, KamodoClient
from kamodo.rpc.proto import to_rpc_expr
import json
from capnp import KjException

def print_rpc(message, indent=0):
    if not isinstance(message, dict):
        message_dict = message.to_dict()
    else:
        message_dict = message
    for k, v in message_dict.items():
        if isinstance(v, dict):
            print_rpc(v)
        else:
            print(k, v)

@kamodofy(units='kg')
def myf(x):
    print('remote f called')
    return x**2-x-1

@kamodofy(units='gram')
def myg(y):
    print('remote g called')
    return y - 1

@kamodofy(units='kg', arg_units=dict(z='cm'))
def myh(z):
    print('remote h called')
    return z**2
    
kserver = Kamodo(f=myf, g=myg, h=myh)
```

```python
kserver
```

```python
import socket
read, write = socket.socketpair()

server = kserver.serve(write)

kclient = KamodoClient(read, verbose=False)
```

## Relay

```python
read2, write2 = socket.socketpair()
```

```python
kclient['f'].meta['hidden_args']
```

```python
import capnp
```

```python
help(capnp.TwoPartyServer.run_forever)
```

```python
kclient['H(x,y)[kg]'] = 'f+g'
```

```python
kclient.H(3,4) ==  3.**2-3-1 + 4-1
```

```python
kclient
```

```python
kclient['mine'] = lambda x: x**2
```

```python
kclient._rpc_funcs
```

```python
kclient._expressions
```

```python
kclient.H(3,4)
```

```python
kclient['H_2(x,y)'] = '2*H(x,y)'
```

```python
kclient._expressions
```

```python
kclient._rpc_funcs
```

```python
kclient
```

```python
assert kclient.H_2(3,4) == 2*(kclient.f(3) + kclient.g(4)/1000)
```

```python
kclient.f(3)
```

```python
kclient.g(2)
```

```python
kclient.H(3,4)
```

```python
assert kclient.H(3,4) == kclient.f(3) + kclient.g(4)/1000
```

```python
kclient['f_2'] = 'x**2-x-1'
```

```python
kclient['H_2(x,y)'] = '2*H'
```

```python
@kamodofy
def myfunc(x):
    print('myfunc called with {}'.format(x))
    return x**2

kclient['H_3'] = myfunc
```

```python
kclient['F_2'] = '2*H_3'
```

```python
kclient.f_2(3)
```

```python
kclient.H(3,4)
```

```python
k2 = Kamodo(H=kclient.H) # creates a new kamodo object loaded from client
```

```python
k2['f(x,y)'] = '2*H'
```

```python
k2.f(3,4)
```

```python
kclient._rpc_funcs
```

```python
result = kclient.H_2(3,4)
result
```

<!-- #region -->
## serve local functions to remote

When we compose with a locally defined function, the remote needs to be able to call the local function in its pipeline.

The function will either be an expression or a lambda:
```python
k = KamodoClient(connection) # registers f_remote
k['f_local'] = kamodofy(lambda x: x**2) # store these in registry
k['g'] = 'f_local+f_remote' # pipelined, will treat these the same
k['h_local'] = 'x**2 -x - 1' # already lambdified, can store in registry
k['g_2'] = '1+f_remote' # executes f_remote and add 1 on the server
k['g_3'] = '1+g_2' # after executing g_2 on server, add 1 on the client
```

We can register each of these in a local dictionary, because we might serve any of them downstream as well as upstream. It may seem wasteful that`g_3` is executed locally, but the user could register a different function if they wanted to avoid that:
```python
k['g_3'] = '2 + f_remote' #executes entirely on server
```

<!-- #endregion -->

```python
kclient.f_2(3) == 3**2 - 3 - 1 # needs to be registered for remote call
```

```python
kclient.f_2
```

## Literals

Kamodo's capnp messages support most core python data types in addition to numpy arrays.

```python
from kamodo.rpc.proto import kamodo_capnp as kcap
```

```python
from kamodo.rpc.proto import array_to_param
```

```python
import numpy as np
```

```python
a = np.linspace(-5,5,12).reshape(3,4)
a
```

* Void: Void
* Boolean: Bool
* Integers: Int8, Int16, Int32, Int64
* Unsigned integers: UInt8, UInt16, UInt32, UInt64
* Floating-point: Float32, Float64
* Blobs: Text, Data
* Lists: List(T)

```python
from kamodo.rpc.proto import array_to_param, param_to_array, kamodo_capnp, from_rpc_literal, test_rpc_literal, to_rpc_literal
import numpy as np
```

```python
test_rpc_literal()
```

```python
literals = [True,
            'hello',
            ['hey', 'there', ['are', 'you', 'listening?']],
            33.3,
            -5,
            None,
           ]

to_rpc_literal(literals).to_dict()
```

```python
to_rpc_literal([True, 1, 1.0]).to_dict()
```

```python
from numpy import ndarray
```

```python
isinstance(np.linspace(-3,3,12), ndarray)
```

```python
to_rpc_literal(np.linspace(-3,3,12).reshape((3,4))).to_dict()
```

```python
result = from_rpc_literal(to_rpc_literal([
    True,
    1,
    70**80,
    [3, 4, 5],
    'json sucks!',
    [bytes(4), 3., 'two', 1, None],
    bytes(2)]))
assert result[-1] == bytes(2)
result
```

<!-- #region -->
# RPC Spec

Kamodo uses capnproto to communicate binary data between functions hosted on different systems. This avoids the need for json serialization and allows for server-side function pipelining while minimizing data transfers.

Kamodo's RPC specification file is located in `kamodo/rpc/kamodo.capnp`:

```sh
{! ../kamodo/rpc/kamodo.capnp !}
```

The above spec allows a Kamodo client (or server) to be implemented in many languages, including [C++](https://capnproto.org/cxx.html), C# (.NET Core), Erlang, Go, Haskell, JavaScript, OCaml, and Rust.

Further reading on capnproto may be found here: 

* [Overview](https://capnproto.org/index.html)
* [Schema language](https://capnproto.org/language.html)
* [RPC](https://capnproto.org/rpc.html)
* Python implementation - [pycapnp](http://capnproto.github.io/pycapnp/quickstart.html)
<!-- #endregion -->
