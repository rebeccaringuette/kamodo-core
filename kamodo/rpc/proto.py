import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()


import capnp
# capnp.remove_import_hook()

kamodo_capnp = capnp.load('{}/kamodo.capnp'.format(current_dir))

from util import get_args, get_defaults
import numpy as np

from sympy import Function

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
    param = kamodo_capnp.Kamodo.Variable.new_message()
    arr_ = np.array(arr)
    if len(arr) > 0:
        param.data = arr_.tobytes()
        param.shape = arr_.shape
        param.dtype = str(arr_.dtype)
    return param




AddRPC = Function('AddRPC')
MulRPC = Function('MulRPC')
PowRPC = Function('PowRPC')

def rpc_expr(expr):
    """Replace expression with RPC functions"""
    if len(expr.args) > 0:
        gather = [rpc_expr(arg) for arg in expr.args]
        if expr.func == Add:
            return AddRPC(*gather)
        if expr.func == Mul:
            return MulRPC(*gather)
        if expr.func == Pow:
            return PowRPC(*gather)
    return expr


class Value(kamodo_capnp.Kamodo.Value.Server):
    "Simple implementation of the Kamodo.Value Cap'n Proto interface."

    def __init__(self, value):
        self.value = value

    def read(self, **kwargs):
        return self.value

def read_value(value):
    """Helper function to asynchronously call read() on a Calculator::Value and
    return a promise for the result.  (In the future, the generated code might
    include something like this automatically.)"""

    return value.read().then(lambda result: result.value)


def evaluate_impl(expression, params=None):
    """Implementation of CalculatorImpl::evaluate(), also shared by
    FunctionImpl::call().  In the latter case, `params` are the parameter
    values passed to the function; in the former case, `params` is just an
    empty list."""

    which = expression.which()
    print('found {}'.format(which))
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
        print('joining promises')
        joinedParams = capnp.join_promises(paramPromises)
        # When the parameters are complete, call the function.
        print('returning result')
        ret = joinedParams.then(lambda vals: func.call(vals)).then(
            lambda result: result.result
        )
        return ret
    else:
        raise ValueError("Unknown expression type: " + which)



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
        return evaluate_impl(expression).then(
            lambda value: setattr(_context.results, "value", Value(value))
        )

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
    
    def getArgs(self, **rpc_kwargs):
        return list(self.args)
        
    def getKwargs(self, **rpc_kwargs):
        if self.verbose:
            print('retrieving kwargs')
        return [dict(name=k, value=array_to_param(v)) for k,v in self.kwargs.items()]
        
    def call(self, args, kwargs, **rpc_kwargs):
        """mimic a pythonic function
        
        raises TypeError when detecting multiple values for argument"""
        
        param_dict = self.kwargs
        
        # insert args
        arg_dict = {}
        for i, value in enumerate(args):
            arg_dict.update({self.args[i]: param_to_array(value)})
        param_dict.update(arg_dict)
        
        # insert kwargs
        for kwarg in kwargs:
            if kwarg.name in arg_dict:
                raise TypeError('multiple values for argument {}, len(args)={}'.format(kwarg.name, len(args)))
            param_dict.update({kwarg.name: param_to_array(kwarg.value)})
        if self.verbose:
            print('serverside function called with {} params'.format(len(param_dict)))
        result = self._func(**param_dict)
        result_param = array_to_param(result)
        return result_param





