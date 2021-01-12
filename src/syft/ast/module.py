# stdlib
import inspect
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast
from ..ast.callable import Callable


def is_static_method(host_object, attr):
    """Test if a value of a class is static method.

    example::

        class MyClass(object):
            @staticmethod
            def method():
                ...

    :param klass: the class
    :param attr: attribute name
    :param value: attribute value
    """
    value = getattr(host_object, attr)

    if not hasattr(host_object, "__mro__"):
        return False

    for cls in inspect.getmro(host_object):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                bound_value = cls.__dict__[attr]
                if isinstance(bound_value, staticmethod):
                    return True
    return False


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    def __init__(
        self,
        client: Optional[Any],
        path_and_name: Optional[str] = None,
        object_ref: Optional[Union["ast.callable.Callable", CallableT]] = None,
        return_type_name: Optional[str] = None,
    ):
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
        )

    def add_attr(
        self,
        attr_name: str,
        attr: Optional[Union[Callable, CallableT]],
        is_static: bool = False,
    ) -> None:
        self.__setattr__(attr_name, attr)

        if is_static is True:
            raise ValueError("MAKE PROPER ERROR SCHEMA")

        if attr is None:
            raise ValueError("MAKE PROPER ERROR SCHEMA")

        self.attrs[attr_name] = attr

    def __call__(
        self,
        path: Optional[List[str]] = None,
        index: int = 0,
    ) -> Optional[Union[Callable, CallableT]]:

        _path: List[str] = (
            path.split(".") if isinstance(path, str) else path if path else []
        )

        resolved = self.attrs[_path[index]](
            path=_path,
            index=index + 1,
        )
        return resolved

    def __repr__(self) -> str:
        out = "Module:\n"
        for name, module in self.attrs.items():
            out += "\t." + name + " -> " + str(module).replace("\t.", "\t\t.") + "\n"

        return out

    def add_path(
        self,
        path: List[str],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[Union[Callable, CallableT]] = None,
    ) -> None:
        if index >= len(path):
            return

        if path[index] not in self.attrs:
            attr_ref = getattr(self.object_ref, path[index])

            if inspect.ismodule(attr_ref):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.module.Module(
                        path_and_name=".".join(path[: index + 1]),
                        object_ref=attr_ref,  # type: ignore
                        return_type_name=return_type_name,
                        client=self.client,
                    ),
                )
            elif inspect.isclass(attr_ref):
                klass = ast.klass.Class(
                    path_and_name=".".join(path[: index + 1]),
                    object_ref=attr_ref,
                    return_type_name=return_type_name,
                    client=self.client,
                )
                self.add_attr(
                    attr_name=path[index],
                    attr=klass,
                )
            elif inspect.isfunction(attr_ref) or inspect.isbuiltin(attr_ref):
                is_static = is_static_method(self.object_ref, path[index])

                self.add_attr(
                    attr_name=path[index],
                    attr=ast.callable.Callable(
                        path_and_name=".".join(path[: index + 1]),
                        object_ref=attr_ref,
                        return_type_name=return_type_name,
                        client=self.client,
                        is_static=is_static,
                    ),
                )
            elif inspect.isdatadescriptor(attr_ref):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.property.Property(
                        path_and_name=".".join(path[: index + 1]),
                        object_ref=attr_ref,
                        return_type_name=return_type_name,
                        client=self.client,
                    ),
                )
            elif index == len(path) - 1:
                if "globals" not in self.attrs:
                    # syft absolute
                    from syft.lib.misc.scope import Scope

                    scope, scope_name = Scope.from_qualname(
                        ".".join(path[:-1]), self.object_ref
                    )
                    path.insert(len(path) - 1, "globals")

                    self.add_attr(
                        attr_name="globals",
                        attr=ast.klass.Class(
                            path_and_name=".".join(path[: index + 1]),
                            object_ref=scope,
                            return_type_name=scope_name,
                            client=self.client,
                        ),
                    )

        attr = self.attrs[path[index]]
        attr.add_path(path=path, index=index + 1, return_type_name=return_type_name)
