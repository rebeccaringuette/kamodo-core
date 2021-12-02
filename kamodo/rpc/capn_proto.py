# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## capnproto interface

import capnp
capnp.remove_import_hook()
addressbook_capnp = capnp.load('addressbook.capnp')

addressbook_capnp.qux

addresses = addressbook_capnp.AddressBook.new_message()

addresses

alice = addressbook_capnp.Person.new_message(name='alice')
alice

alice.name

people = addresses.init('people', 2) # don't call init more than once!

people[0] = alice

people[1] = addressbook_capnp.Person.new_message(name='bob')

alicePhone = alice.init('phones', 1)[0]

# ## enum type

alicePhone.type == 'mobile'

try:
    alicePhone.type = 'pager'
except AttributeError as m:
    print(m)

# ## unions

# unions are like enum structs. employment has type union, making it unique among a (fixed?) set of options.

alice.employment.which()

try:
    alice.employment.spaceship = 'Enterprise'
except AttributeError as m:
    print(m)

alice.employment.school = 'Rice'
print(alice.employment.which())

alice.employment.unemployed = None
print(alice.employment.which())

# ## i/o
# The whole point of rpc is that we can communicate in binary.

with open('example.bin', 'wb') as f:
    addresses.write(f)

# cat example.bin

with open('example.bin', 'rb') as f:
    addresses = addressbook_capnp.AddressBook.read(f)

first_person = addresses.people[0]

first_person.name

type(first_person)

employment = first_person.employment.which()
print('{} is employed at {}'.format(first_person.name, employment))
getattr(first_person.employment, employment)

# ## Dict/list

addresses.to_dict()

alice

# ## builders vs readers
# When you create a message, you are making a builder. When you read a message, you're using a reader.

type(alice)

type(addresses.people[0])

# builders have a to_bytes method

alice.to_bytes()

addressbook_capnp.Person.from_bytes(alice.to_bytes())

# ## packed format
#
# The binary data can be sent in compressed format

addressbook_capnp.Person.from_bytes_packed(alice.to_bytes_packed())

# ## RPC
#
# Specification here https://capnproto.org/rpc.html
#
# The key idea is minimizing the number of trips to/from the server by chaining dependent calls. This fits really well with kamodo's functional style, where the user is encouraged to use function composition in their pipelines! We want to be able to leverage these capabilities in our `KamodoRPC` class.

# ### Calculator test
#
# The calculator spec is located in `calculator.capnp` and is copied from the pycapnp repo.
#
# Run the calculator server in a separate window before executing these cells.
#
# `python calculator_server.py 127.0.0.1:6000`

calculator_capnp = capnp.load('calculator.capnp')
client = capnp.TwoPartyClient('127.0.0.1:6000')

# ### bootstrapping
# There could be many interfaces defined within a given service. The client's `bootstrap` method will get the interface marked for bootstrapping by server.
#
# First bootstrap the Calculator interface

calculator = client.bootstrap().cast_as(calculator_capnp.Calculator)

calculator2 = client.bootstrap().cast_as(calculator_capnp.Calculator)

# The server defines which interface to bootstrap: `TwoPartyServer(address, bootstrap=CalculatorImpl()`.

# ### methods
# Ways to call an RPC method

# +
request = calculator.evaluate_request()
request.expression.literal = 123
eval_promise = request.send()

# result = eval_promise.wait().value.read().wait() # blocking?
read_result = eval_promise.then(lambda ret: ret.value.read()).wait() # chained
read_result.value
# -

read_result

# You may also interogate available rpc methods:

calculator.schema.method_names

calculator.schema.methods

# ### test rpc
# * can test rpc with socket pair:
#
# ```python
#
# class ServerImpl():
#     ...
#
# read, write = socket.socketpair()
#
# _ = capnp.TwoPartyServer(write, bootstrap=ServerImpl())
# client = capnp.TwoPartyClient(read)
# ```
#

# ### Type Ids
#
# To generate file ids, make sure you have capnp command line tool installed:
# ```sh
# conda install -c conda-forge capnp
# capnp id # generates unique file id
# capnp compile -ocapnp calculator.capnp # returns a schema filled in with ids for all new types
# ```
#
# The unique type identifiers aid in backward compatibility and schema flexibility.
#
# capnproto schema language reference https://capnproto.org/language.html

# ## RPC Parameters
# These variables can be created by the server/client and wrap numpy arrays.

# +
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')
import numpy as np

def class_name(obj):
    """get fully qualified class name of object"""
    return ".".join([obj.__class__.__module__, obj.__class__.__name__])

def param_to_array(param):
    """convert from parameter to numpy array
    assume input is numpy binary
    """
    if len(param.data) > 0:
        return np.frombuffer(param.data).reshape(param.shape)
    else:
        return np.array([])

def array_to_param(arr):
    """convert an array to an rpc parameter"""
    param = kamodo_capnp.Kamodo.Variable.new_message()
    if len(arr) > 0:
        param.data = arr.tobytes()
        param.shape = arr.shape
        param.dtype = class_name(arr)
    return param


# -

a = np.linspace(-5,5,12).reshape(3,4)

a

b = array_to_param(a)

b.to_dict()

# ## RPC Functions
#
# These functions execute on the server.

# +
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

# import kamodo_capnp
# -

import numpy as np


class Poly(kamodo_capnp.Kamodo.Function.Server):
    def __init__(self):
        pass
        
    def call(self, params, **kwargs):
        if len(params) == 0:
            return kamodo_capnp.Kamodo.Variable.new_message()
        print('serverside function called with {} params'.format(len(params)))
        param_arrays = [param_to_array(_) for _ in params]
        x = sum(param_arrays)
        result = x**2 - x - 1
        result_ = array_to_param(result)
        return result_


# Set up a client/server socket for testing.

import socket
read, write = socket.socketpair()

# instantiate a server with a Poly object

server = capnp.TwoPartyServer(write, bootstrap=Poly())

# instantiate a client with bootstrapping
#
# > capabilities are intrinsically dynamic, and they hold no run time type information, so we need to pick what interface to interpret them as.

client = capnp.TwoPartyClient(read)
# polynomial implementation lives on the server
polynomial = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)

# +
poly_promise = polynomial.call([b,b,b])

# evaluate ...

response = poly_promise.wait()
# -

param_to_array(response.result) # (sum(b))**2 - sum(b) - 1


class FunctionRPC:
    def __init__(self):
        self.func = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)
    
    def __call__(self, params):
        params_ = [array_to_param(_) for _ in params]
        func_promise = self.func.call(params_)
        # evaluate
        response = func_promise.wait().result
        return param_to_array(response)


serverside_function = FunctionRPC()

a

serverside_function([a, a, a])

# ## Function groups

import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

kamodo_capnp.Kamodo.Function

a = np.linspace(-1,1,10)
b = array_to_param(a)
b.to_dict()

param = kamodo_capnp.Kamodo.Parameter.new_message(symbol='x', value=b).to_dict()
param

kamodo_capnp.Kamodo.Field.new_message(
            symbol='P_n',
            func=Poly(),
            defaults=[dict(symbol='x', value=b)],
        ).to_dict()

kamodo_capnp.Kamodo.Field.new_message(
            symbol='P_n',
            func=Poly(),
            defaults=[dict(param='x', value=b)],
        ).to_dict()

field = kamodo_capnp.Kamodo.Field.new_message(
            symbol='P_n',
            func=Poly(),
            defaults=[dict(symbol='x', value=b)],
        )   

field.to_dict()

field.defaults[0].symbol

field.defaults[0].value.to_dict()

import forge

defaults = {}
for _ in field.defaults:
    defaults[_.symbol] = param_to_array(_.value)

defaults

# +
from kamodo.util import construct_signature


import socket
read, write = socket.socketpair()

from kamodo import Kamodo, kamodofy

b = array_to_param(a)

class KamodoServer(kamodo_capnp.Kamodo.Server):
    def __init__(self):
        field = kamodo_capnp.Kamodo.Field.new_message(
            symbol='P_n',
            func=Poly(),
            defaults=[dict(symbol='x', value=b)],
        )        
        self.fields = [field]
        
        
    def getFields(self, **kwargs):
        return self.fields
    
server = capnp.TwoPartyServer(write, bootstrap=KamodoServer())
    
class KamodoClient(Kamodo):
    def __init__(self, client, **kwargs):
        self._client = client.bootstrap().cast_as(kamodo_capnp.Kamodo)
        self._rpc_fields = self._client.getFields().wait().fields
        
        super(KamodoClient, self).__init__(**kwargs)
        
        for field in self._rpc_fields:
            self.register_rpc(field)
            
    def register_rpc(self, field):
        print(field.to_dict())
        defaults = {}
        for _ in field.defaults:
            defaults[_.symbol] = param_to_array(_.value)
            
        @kamodofy
        @forge.sign(*construct_signature(**defaults))
        def remote_func(**kwargs):
            # params must be List(Variable) for now
            params = [array_to_param(v) for k,v in kwargs.items()]
            response = field.func.call(params=params).wait().result
            return param_to_array(response)

        self[field.symbol] = remote_func
        
client = capnp.TwoPartyClient(read)
        
kclient = KamodoClient(client)
kclient
# -

kclient.P_n(np.linspace(-5,5,33))

# ## String Sanitizing

from asteval import Interpreter

aeval = Interpreter()

aeval('x=3')
aeval('1+x')

aeval.symtable['sum']

# ## RPC expressions
#
# Wrap sympy expressions with placeholder alegebraic calls (to be executed on server)

# +
from sympy import Function, sympify
from sympy import Add, Mul, Pow
from functools import reduce
from operator import mul, add, pow

AddRPC = Function('AddRPC')
MulRPC = Function('MulRPC')
PowRPC = Function('PowRPC')


def rpc_expr(expr):
    if len(expr.args) > 0:
        gather = [rpc_expr(arg) for arg in expr.args]
        if expr.func == Add:
            return AddRPC(*gather)
        if expr.func == Mul:
            return MulRPC(*gather)
        if expr.func == Pow:
            return PowRPC(*gather)
    return expr


expr_ = sympify('30*a*b + c**2+sin(c)')
rpc_expr(expr_)


def add_impl(*args):
    print('computing {}'.format('+'.join((str(_) for _ in args))))
    return reduce(add, args)

def mul_impl(*args):
    print('computing {}'.format('*'.join((str(_) for _ in args))))
    return reduce(mul, args)

def pow_impl(base, exp):
    print('computing {}^{}'.format(base, exp))
    return pow(base,exp)


# -

add_impl(3,4,5)

mul_impl(3,4,5)

pow_impl(3,4)

func_impl = dict(AddRPC=add_impl,
                 MulRPC=mul_impl,
                 PowRPC=pow_impl)

from kamodo import Kamodo
from sympy import lambdify
from kamodo.util import sign_defaults


# +
class KamodoClient(Kamodo):
    def __init__(self, server, **kwargs):
        self._server = server
        super(KamodoClient, self).__init__(**kwargs)
        
    def vectorize_function(self, symbol, rhs_expr, composition):
        """lambdify the input expression using server-side promises"""
        print('vectorizing {} = {}'.format(symbol, rhs_expr))
        print('composition keys {}'.format(list(composition.keys())))
        func = lambdify(symbol.args,
                        rpc_expr(rhs_expr),
                        modules=[func_impl, 'numpy', composition])
        signature = sign_defaults(symbol, rhs_expr, composition)
        return signature(func)
    
kamodo = KamodoClient('localhost:8050')
kamodo['f[cm]'] = 'x**2-x-1'
kamodo['g'] = 'f**2'
kamodo['h[km]'] = 'f'
kamodo
# -

kamodo.f(3)

assert kamodo.f(3) == 3**2 - 3 - 1

kamodo

# ## Serverside Algebra

# +
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

import socket
read, write = socket.socketpair()

# +
from operator import mul, add, pow
from functools import reduce

# def add_impl(*args):
#     print('computing {}'.format('+'.join((str(_) for _ in args))))
#     return reduce(add, args)

# def mul_impl(*args):
#     print('computing {}'.format('*'.join((str(_) for _ in args))))
#     return reduce(mul, args)

# def pow_impl(base, exp):
#     print('computing {}^{}'.format(base, exp))
#     return pow(base,exp)

class AddImpl(kamodo_capnp.Kamodo.Function.Server):
    def call(self, params, **kwargs):
        result = reduce(add, param_arrays)
        return array_to_param(result)


class Algebra(kamodo_capnp.Kamodo.Server):
    def __init__(self):
        self.add = AddImpl()


# -

server = capnp.TwoPartyServer(write, bootstrap=Algebra())

client = capnp.TwoPartyClient(read)


class ClientSideFunction:
    def __init__(self, client, op):
        self.kamodo = client.bootstrap().cast_as(kamodo_capnp.Kamodo)
    
    def __call__(self, *params):
        params_ = [array_to_param(_) for _ in params]
        print('client passing params to server')
        func_promise = self.kamodo.Algebra.add.call(params_)
        # evaluate
        response = func_promise.wait().result
        return param_to_array(response)


f = ClientSideFunction(client, 'add')

import numpy as np

a = np.linspace(-1,1,15)

f(a,a,a,a)

# ## Kamodo Fields
#
# Define fields available for remote call

# +
from kamodo import Kamodo
import capnp
# import kamodo_capnp

class KamodoRPCImpl(kamodo_capnp.Kamodo.Server):
    """Interface class for capnp"""
    def __init__(self):
        pass
    
    def getFields(self, **kwargs):
        """
        Need to return a list of fields
          struct Field {
            symbol @0 :Text;
            func @1 :Function;
          }
        """

        f = kamodo_capnp.Kamodo.Field.new_message(symbol='f', func=Poly())
        return [f]


# -

read, write = socket.socketpair()

write = capnp.TwoPartyServer(write, bootstrap=KamodoRPCImpl())

client = capnp.TwoPartyClient(read)
kap = client.bootstrap().cast_as(kamodo_capnp.Kamodo)

fields = kap.getFields().wait().fields

param_to_array(fields[0].func.call([
    array_to_param(a)]).wait().result)


class KamodoRPC(Kamodo):
    def __init__(self, read_url=None, write_url=None, **kwargs):
        
        if read_url is not None:
            self.client = capnp.TwoPartyClient(read_url)
        if write_url is not None:
            self.server = capnp.TwoPartyServer(write_url,
                                               bootstrap=kamodo_capnp.Kamodo())
        super(Kamodo, self).__init__(**kwargs)


from kamodo import reserved_names

from sympy.abc import _clash # gathers reserved symbols
