import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()


import capnp
# capnp.remove_import_hook()

kamodo_capnp = capnp.load('{}/kamodo.capnp'.format(current_dir))

from util import get_args, get_defaults
import numpy as np

from sympy import Function, Symbol
from sympy import Add, Mul, Pow
from sympy.core.numbers import Float, Integer

from functools import reduce

def rpc_map_to_dict(rpc_map, callback = None):
    if callback is not None:
        return {_.key: callback(_.value) for _ in rpc_map.entries}
    else:
        return {_.key: _.value for _ in rpc_map.entries}
        

def rpc_dict_to_map(d, callback = None):
    if callback is not None:
        entries=[dict(key=k, value=callback(v)) for k,v in d.items()]
    else:
        entries=[dict(key=k, value=v) for k, v in d.items()]
    return dict(entries=entries)
    
# def class_name(obj):
#     """get fully qualified class name of object"""
#     return ".".join([obj.__class__.__module__, obj.__class__.__name__])

def param_to_array(param):
    """convert from parameter to numpy array
    assume input is numpy binary
    """
    if len(param.data) > 0:
        return np.frombuffer(param.data, dtype=param.dtype).reshape(param.shape)
    else:
        return np.array([], dtype=param.dtype)

def array_to_param(arr):
    """convert an array to an rpc parameter"""
    param = kamodo_capnp.Kamodo.Array.new_message()
    arr_ = np.array(arr)
    if len(arr) > 0:
        param.data = arr_.tobytes()
        param.shape = arr_.shape
        param.dtype = str(arr_.dtype)
    return param


def from_rpc_literal(literal):
    """unwrap a literal"""
    which = literal.which()
    # print('unwrapping literal {}'.format(which))
    if which == 'void':
        return None
    elif which == 'bool':
        return bool(literal.bool)
    elif which == 'array':
        return param_to_array(literal.array)
    elif which == 'text':
        return str(literal.text)
    elif which == 'data':
        return literal.data
    elif which == 'list':
        return [from_rpc_literal(lit) for lit in literal.list]
    elif which in ('int8', 'int16', 'int32', 'int64',
                   'uint8', 'uint16', 'uint32', 'uint64',
                   'float32', 'float64'):
        return getattr(np, which)(getattr(literal, which))
    elif which == 'int':
        return int(literal.int)
    else:
        raise NotImplementedError('Unknown type {}'.format(which))

def unique_type(alist):
    if len(alist) > 0:
        atype = type(alist[0])
        return all(isinstance(_, atype) for _ in alist)
    return True

def to_rpc_literal(value):
    """
    void @0 :Void;
    bool @1 :Bool;
    int8 @2 :Int8;
    int16 @3 :Int16;
    int32 @4 :Int32;
    int64 @5 :Int64;
    uint8 @6 :UInt8;
    uint16 @7 :UInt16;
    uint32 @8 :UInt32;
    uint64 @9 :UInt64;
    float32 @10 :Float32;
    float64 @11 :Float64;
    text @12 :Text;
    data @13 :Data;
    list @14 :List(Literal);
    array @15 :Array;
    int @16 :Text;
    listint64 @17 :List(Int64);
    listfloat64 @18 :List(Float64);
    """
    if isinstance(value, bytes):
        which = 'data'
    elif isinstance(value, np.ndarray):
        which = 'array'
        value = array_to_param(value)
    elif value is None:
        which = 'void'
    elif isinstance(value, bool):
        which = 'bool'
        value = bool(value)
    elif isinstance(value, int):
        # will encode as text (too many types of ints to choose from)
        which = 'int'
        value = str(value)
    elif isinstance(value, float):
        # python standard float is C double
        which = 'float64'
        value = float(value) # cast np.float as float
    elif isinstance(value, str):
        which = 'text'
    elif isinstance(value, list):
        if unique_type(value):
            pass

        which = 'list'
        value = [to_rpc_literal(_) for _ in value]
    else:
        which = type(value).__name__
    try:
        return kamodo_capnp.Kamodo.Literal(**{which: value})
    except Exception as m:
        raise NotImplementedError('{} type not yet supported: {}\n{}'.format(
            which, value.to_dict(), m))

def test_rpc_literal():
    a = np.linspace(-5,5,12).reshape(3,4)
    lit_check = dict(
        void=None,
        bool=True,
        int8=-2,
        int16=-4,
        int32=-8,
        int64=-16,
        uint8=2,
        uint16=4,
        uint32=6,
        uint64=11,
        float32=3,
        float64=3,
        array=array_to_param(a),
        text='hello there',
        int='30000333',
        list=[
            dict(float32=3),
            dict(list=[dict(list=[
                dict(bool=True),
                dict(array=array_to_param(a)),
                ])])
            ]
        )
    lit = kamodo_capnp.Kamodo.Literal(list = [{k: v} for k,v in lit_check.items()])

    for _ in from_rpc_literal(lit):
        print('{}: {}'.format(type(_).__name__, _))

# AddRPC = Function('AddRPC')
# MulRPC = Function('MulRPC')
# PowRPC = Function('PowRPC')

# def rpc_expr(expr):
#     """Replace expression with RPC functions"""
#     if len(expr.args) > 0:
#         gather = [rpc_expr(arg) for arg in expr.args]
#         if expr.func == Add:
#             return AddRPC(*gather)
#         if expr.func == Mul:
#             return MulRPC(*gather)
#         if expr.func == Pow:
#             return PowRPC(*gather)
#     return expr


class Value(kamodo_capnp.Kamodo.Value.Server):
    "Simple implementation of the Kamodo.Value Cap'n Proto interface."

    def __init__(self, value):
        self.value = value

    def read(self, **kwargs):
        return to_rpc_literal(self.value)

def read_value(value):
    """Helper function to asynchronously call read() on a Calculator::Value and
    return a promise for the result.  (In the future, the generated code might
    include something like this automatically.)"""

    return value.read().then(lambda result: result.value)


def evaluate_impl(expression, params=None):
    """
    Borrows heavily from CalculatorImpl::evaluate()"""

    which = expression.which()
    if which == "literal":
        return capnp.Promise(expression.literal)
    elif which == "store":
        return read_value(expression.store)
    elif which == "parameter":
        assert expression.parameter < len(params)
        return capnp.Promise(params[expression.parameter])
    elif which == "call":
        call = expression.call
        func = call.function

        # Evaluate each parameter.
        paramPromises = [evaluate_impl(param, params) for param in call.params]
        joinedParams = capnp.join_promises(paramPromises)
        # When the parameters are complete, call the function.
        ret = joinedParams.then(
            lambda vals: func.call(vals)).then(
            lambda result: result.result
        )
        return ret
    else:
        raise NotImplementedError("Unknown expression type: " + which)


class KamodoRPC(kamodo_capnp.Kamodo.Server):
    def __init__(self, **fields):
        self.fields = fields

        self.math = dict(
            AddRPC=FunctionRPC(lambda *params: np.add(*params)),
            MulRPC=FunctionRPC(lambda *params: np.multiply(*params)),
            PowRPC=FunctionRPC(lambda base_, exp_: np.power(base_, exp_))
            )

    def getFields(self, **kwargs):
        # getFields @0 () -> (fields :Map(Text, Field));
        return rpc_dict_to_map(self.fields)


    def getMath(self, **kwargs):
        # getMath @1 () -> (math :Map(Text, Function));
        return rpc_dict_to_map(self.math)


    def evaluate(self, expression, _context, **kwargs):
        # evaluate @2 (expression: Expression) -> (value: Value);
        evaluated = evaluate_impl(expression)
        result = evaluated.then(
            lambda value: setattr(
                _context.results,
                "value",
                Value(from_rpc_literal(value)))
            )
        return result

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, field):
        self.fields[key] = field


class FunctionRPC(kamodo_capnp.Kamodo.Function.Server):
    def __init__(self, func, verbose=False):
        """Converts a function to RPC callable"""
        self._func = func
        self.verbose = verbose
        self.args = get_args(self._func)
        self.kwargs = get_defaults(self._func)

    def __repr__(self):
        return "FunctionRPC({})".format(self._func.__name__)

    def getArgs(self, **rpc_kwargs):
        """getArgs @1 () -> (args :List(Text));"""
        return list(self.args)
        
    def getKwargs(self, **rpc_kwargs):
        """getKwargs @2 () -> (kwargs: List(Argument));"""
        if self.verbose:
            print('retrieving kwargs')
        return [dict(name=k, value=to_rpc_literal(v)) for k,v in self.kwargs.items()]
        
    def call(self, args, kwargs, **rpc_kwargs):
        """call @0 (args :List(Literal), kwargs :List(Argument)) -> (result: Literal);

        mimic a pythonic function
        raises TypeError when detecting multiple values for argument"""
        
        param_dict = self.kwargs
        
        # insert args
        arg_dict = {}
        for i, value in enumerate(args):
            arg_dict.update({self.args[i]: from_rpc_literal(value)})
        param_dict.update(arg_dict)
        
        # insert kwargs
        for kwarg in kwargs:
            if kwarg.name in arg_dict:
                raise TypeError('multiple values for argument {}, len(args)={}'.format(kwarg.name, len(args)))
            param_dict.update({kwarg.name: from_rpc_literal(kwarg.value)})
        if self.verbose:
            print('serverside function called with {} params'.format(len(param_dict)))
        result = self._func(**param_dict)
        result_param = to_rpc_literal(result)
        return result_param


class AddRPC(kamodo_capnp.Kamodo.Function.Server):
    def getArgs(self, **rpc_kwargs):
        return []

    def getKwargs(self, **rpc_kwargs):
        return []

    def call(self, args, kwargs, **rpc_kwargs):
        """call @0 (args :List(Literal), kwargs :List(Argument)) -> (result: Literal)"""
        args_ = [from_rpc_literal(arg) for arg in args]
        result = reduce(lambda a, b: a+b, args_)
        return to_rpc_literal(result)

class MulRPC(kamodo_capnp.Kamodo.Function.Server):
    def getArgs(self, **rpc_kwargs):
        return []

    def getKwargs(self, **rpc_kwargs):
        return []

    def call(self, args, kwargs, **rpc_kwargs):
        """call @0 (args :List(Literal), kwargs :List(Argument)) -> (result: Literal)"""
        args_ = [from_rpc_literal(arg) for arg in args]
        result = reduce(lambda a, b: a*b, args_)
        return to_rpc_literal(result)

add_rpc = AddRPC()
mul_rpc = MulRPC()
pow_rpc = FunctionRPC(lambda a, b: a**b)

math_rpc = {Add: add_rpc, Mul:mul_rpc, Pow: pow_rpc}


def to_rpc_expr(expr, math_rpc=math_rpc, **kwargs):
    """takes a sympy expression with kwargs and returns
    an RPC expression ready for evaluation
    math_rpc is a dictionary mapping <func symbol> -> <rpc function>
    
    math_rpc = {Add: add_rpc, Mul: mul_rpc, Pow: pow_rpc}
    
    Note: math_rpc can contain a mix of client and server-defined rpc funcions
    """
    message = dict()
    if len(expr.args) > 0:
        params = [to_rpc_expr(arg, math_rpc, **kwargs) for arg in expr.args]
        message['call'] = dict(params=params)
        if expr.func in math_rpc:
            message['call']['function'] = math_rpc[expr.func]
        elif str(expr.func) in kwargs:
            message['call']['function'] = kwargs[str(expr.func)]
        else:
            raise NotImplementedError("{}".format(expr.func))
    elif isinstance(expr, Float):
        message['literal'] = to_rpc_literal(float(expr))
    elif isinstance(expr, Integer):
        message['literal'] = to_rpc_literal(int(expr))
    elif isinstance(expr, Symbol):
        sym = str(expr)
        if sym in kwargs:
            message['literal'] = to_rpc_literal(kwargs[sym])
        else:
            raise TypeError('Expression missing required argument {}'.format(sym))
    else:
        raise NotImplementedError(expr)
    return kamodo_capnp.Kamodo.Expression(**message)


