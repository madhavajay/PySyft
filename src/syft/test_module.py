# stdlib
from enum import Enum

global_value = 5


def global_function():
    return global_value


class A:
    __slots__ = ["_private_attr"]

    static_attr = 4

    def __init__(self):
        self._private_attr = 5.5

    def test_method(self):
        return 0

    @property
    def test_property(self):
        return self._private_attr

    @staticmethod
    def static_method():
        return True

    @test_property.setter
    def test_property(self, value):
        self._private_attr = value


class B(Enum):
    Car = 1
    Cat = 2
    Dog = 3
