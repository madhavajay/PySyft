# stdlib
from enum import Enum

global_value = 5


def global_func():
    return global_value


class A:
    __slots__ = ["_private_attr"]

    def __init__(self):
        self._private_attr = 5.5

    def test_method(self):
        return 0

    @property
    def test_property(self):
        return self._private_attr

    @test_property.setter
    def test_property(self, value):
        self._test_property = value

    @staticmethod
    def static_method():
        return True

    @classmethod
    def class_method(cls):
        pass


class B(Enum):
    Car = 1
    Cat = 2
    Dog = 3
