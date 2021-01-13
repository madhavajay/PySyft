# stdlib
import inspect
import sys
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

scopes_cache = {}


def generate_property(module: Any, property_name: str) -> property:
    def prop(self: Any) -> Any:
        return getattr(module, property_name)

    def getter(self: Any) -> Any:
        return getattr(module, property_name)

    def setter(self: Any, elem: Any) -> None:
        setattr(module, property_name, elem)

    def deleter(self: Any) -> None:
        obj = getattr(module, property_name)
        del obj

    generated_property = property(prop)
    generated_property = generated_property.setter(setter)
    generated_property = generated_property.getter(getter)
    generated_property = generated_property.deleter(deleter)

    return generated_property


def generate_func(module: ModuleType, method_name: str) -> Callable:
    def proxy_func(*args: Any, **kwargs: Any) -> Any:
        object = getattr(module, method_name)
        return object(*args, **kwargs)

    return proxy_func


class Scope(type):
    def __new__(cls, name: str, bases: tuple, dct: Dict[str, Any]) -> Any:
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def from_qualname(module_path: str, module) -> Tuple["Scope", str]:
        path = module_path.split(".")
        functions: Dict[str, Any] = {}

        print(module_path)
        for method_name in dir(module):
            target_obj = getattr(module, method_name)

            if callable(target_obj):
                if inspect.isfunction(target_obj):
                    functions[method_name] = generate_func(module, method_name)
            else:
                functions[method_name] = generate_property(module, method_name)

        name = "".join(path) + "_global_scope"
        functions["__qualname__"] = "syft.lib.misc.scope." + name
        functions["__name__"] = name

        type = Scope(name, tuple(), functions)

        globals()[name] = type
        scopes_cache[name] = type

        return type, type.__qualname__
