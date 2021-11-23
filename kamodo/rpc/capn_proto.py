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

result.value

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
