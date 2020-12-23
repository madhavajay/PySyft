"""
The following test suit serves as a set of examples of how to integrate different classes
into our AST and use them.
"""
# syft absolute
import syft
from syft.ast.globals import Globals
from syft.lib import create_lib_ast
from syft.lib import registered_callbacks

# syft relative
from syft import test_module


def create_AST(client):
    ast = Globals(client)

    methods = [
        ("test_module.A", "test_module.A"),
        ("test_module.A.test_method", "syft.lib.python.Int"),
        ("test_module.A.test_property", "syft.lib.python.Float"),
        ("test_module.A._private_attr", "syft.lib.python.Float"),
        ("test_module.A.static_method", "syft.lib.python.Float"),
        ("test_module.A.static_attr", "syft.lib.python.Int"),
        ("test_module.B.Car", "test_module.B"),
        # ("test_module.global_value", "syft.lib.python.Int"),
        # ("test_module.global_function", "syft.lib.python.Int"),
    ]

    for method, return_type in methods:
        ast.add_path(
            path=method, framework_reference=test_module, return_type_name=return_type
        )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast


def get_custom_client():
    registered_callbacks["test_module"] = create_AST
    syft.lib_ast = create_lib_ast(None)
    alice = syft.VirtualMachine(name="alice")
    alice_client = alice.get_root_client()
    return alice_client


def test_method():
    client = get_custom_client()
    a_ptr = client.test_module.A()
    result_ptr = a_ptr.test_method()

    a = test_module.A()
    result = a.test_method()

    assert result == result_ptr.get()

def test_property_get():
    client = get_custom_client()
    a_ptr = client.test_module.A()
    result_ptr = a_ptr.test_property

    a = test_module.A()
    result = a.test_property

    assert result == result_ptr.get()


def test_property_set():
    value_to_set = 7.5
    client = get_custom_client()

    a_ptr = client.test_module.A()
    a_ptr.test_property = value_to_set
    result_ptr = a_ptr.test_property

    a = test_module.A()
    a.test_property = value_to_set
    result = a.test_property

    assert result == result_ptr.get()


def test_slot_get():
    client = get_custom_client()

    a_ptr = client.test_module.A()
    result_ptr = a_ptr._private_attr

    a = test_module.A()
    result = a._private_attr

    assert result == result_ptr.get()


def test_slot_set():
    value_to_set = 7.5
    client = get_custom_client()

    a_ptr = client.test_module.A()
    a_ptr._private_attr = value_to_set
    result_ptr = a_ptr._private_attr

    a = test_module.A()
    a._private_attr = value_to_set
    result = a._private_attr

    assert result == result_ptr.get()


def test_global_function():
    client = get_custom_client()

    globals = client.test_module.globals()
    result_ptr = globals.global_function()
    result = test_module.global_function()

    assert result == result_ptr.get()



def test_global_attribute_get():
    client = get_custom_client()

    globals = client.test_module.globals()
    result_ptr = globals.global_value

    result = test_module.global_value

    assert result == result_ptr.get()



def test_global_attribute_set():
    client = get_custom_client()

    globals = client.test_module.globals()
    result_ptr = globals.global_value
    result = test_module.global_value

    assert result == result_ptr.get()


def test_static_method():
    client = get_custom_client()

    result_ptr = client.test_module.A.static_method()
    result = test_module.A.static_method()
    print(result)
    assert result == result_ptr.get()

test_static_method()

def test_static_attribute_get():
    client = get_custom_client()

    result_ptr = client.test_module.A.static_attr
    result = test_module.A.static_attr

    assert result == result_ptr.get()

def test_static_attribute_set():
    value_to_set = 5
    client = get_custom_client()

    client.test_module.A.static_attr = value_to_set
    result_ptr = client.test_module.A.static_attr

    test_module.A.static_attr = value_to_set
    result = test_module.A.static_attr

    assert result == result_ptr.get()



def test_enum():
    client = get_custom_client()

    result_ptr = client.test_module.B.Car
    result = test_module.B.Car

    assert result == result_ptr.get()

def test_dynamic_attribute():
    pass
