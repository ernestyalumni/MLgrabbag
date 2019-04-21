# -*- coding: utf-8 -*-
"""
@file ToURL.py
@ref https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/alphavantage.py
"""
from urllib.request import Request
import urllib.request

def handle_API_call(url):
    return Request(url)

def response_from_API_call(url):
    return urllib.request.urlopen(url)

#def handle_function_arguments(function):
"""
@fn handle_function_arguments
@brief Wraps inspect.getfullargspec.

@details inspect.getfullargspec(func) gets the names and default values of
a Python function's parameters.

A named tuple is returned by inspect.getfullargspec:
    
FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults,
            annotations)

args is a list of positional parameter names.
varargs is the name of the * parameter or None if arbitrary positional
arguments aren't accepted.
varkw is the name of the ** parameter or None if arbitrary keyword
arguments aren't accepted.
defaults is an n-tuple of default argument values corresponding to the last
n positional parameters, or None if there are no such defaults defined.
kwonlyargs is a list of keyword-only parameter names in declaration order.
kwonlydefaults is a dictionary mapping parameter names from kwonlyargs to
the default values used if no argument is supplied.
annotations is a dictionary mapping parameter names to annotations.

@ref https://stackoverflow.com/questions/28221885/inspect-getfullargspec-in-the-python
e.g.
def example(a:int, b=1, *c, d, e=2, **f) -> str:
    pass
    
print(inspect.getfullargspec(example))

# prints out
FullArgSpec(args=['a', 'b'], varargs='c', varkw='f', defaults=(1,), \
kwonlyargs=['d', 'e'], kwonlydefaults={'e': 2}, \
annotations={'a': <class 'int'>, 'return': <class 'str'>}
"""
 #   full_argument_specification = inspect.getfullargspec(function)

    