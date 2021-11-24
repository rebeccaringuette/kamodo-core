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

# ## KamodoRPC

# +
import capnp
# capnp.remove_import_hook()
# kamodo_capnp = capnp.load('kamodo.capnp')

import kamodo_capnp
# -

import numpy as np


# +
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


# -

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

a = np.linspace(-5,5,12).reshape(3,4)

a

b = array_to_param(a)

b.to_dict()

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

Poly

# +
from kamodo import Kamodo
import capnp
import kamodo_capnp

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


# h(x,y) = f_server1(x) + g_server1(y)

from sympy import lambdify
from sympy.abc import a,b,c


# +
def myadd(*args):
    print("hey {}, I'll make you a promise".format(args))
    return sum(args)

def mySymbol(symbol):
    print("hey {}, I'll hold onto you".format(symbol))
    return symbol

expr = a+b+c

def remote_expr(expr):
    """mock execution on remote server"""
    func = lambdify(args = expr.args,
                    expr = srepr(a+b+c).replace(
                        'Add(', 'myadd(').replace(
                        'Symbol(', 'mySymbol('),
                   modules = dict(myadd=myadd,
                                  mySymbol=mySymbol))
    return func

remote_expr(a+b+c)(3,4,5+2)
# -


