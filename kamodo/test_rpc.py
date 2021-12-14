"""
Tests for rpc interface
"""

import pytest

from kamodo import Kamodo

import capnp
import socket


# pytest.raises(NameError)

def test_register_remote():
	k = Kamodo(f='x**2-x-1')

	read, write = socket.socketpair()

	kserver = k.server(write)
        
	k2 = Kamodo()
	client = k2.client(read)

	assert k2.f([3])[0] == 3**2-3-1