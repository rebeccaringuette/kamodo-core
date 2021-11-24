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
    return np.frombuffer(param.data).reshape(param.shape)

def array_to_param(arr):
    param = kamodo_capnp.Kamodo.Variable.new_message()
    param.data = arr.tobytes()
    param.shape = arr.shape
    param.dtype = class_name(arr)
    return param


# -

class poly(kamodo_capnp.Kamodo.Function.Server):
    def call(self, fields, **kwargs):
        print('poly called')
        x = param_to_array(fields[0])
        result = x**2 - x - 1
        result_ = array_to_param(result)
        return result_



# Set up the server-side poly function.

# +
import socket
read, write = socket.socketpair()

server = capnp.TwoPartyServer(write, bootstrap=poly)

# +
client = capnp.TwoPartyClient(read)

polynomial = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)
# -

a = np.linspace(-5,5,12).reshape(3,4)

b = array_to_param(a)

result_promise = polynomial.call([b])

result_promise.wait()

param_to_array(result.wait())

# +
remote = cap.foo(i=5)
response = remote.wait()

assert response.x == "125"
# -

kap = kamodo_capnp


class Server(kamodo_capnp.Kamodo.Server):
    def foo(self, i, j, **kwargs):
        return str(i * 5 + self.val)


kamodo_capnp.Variable


