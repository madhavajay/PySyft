# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import Optional
from typing import Tuple
from typing import Union

# syft relative
from .. import ast
from ..core.node.common.action.get_or_set_property_action import GetOrSetPropertyAction


class Property(ast.attribute.Attribute):
    def __init__(
        self,
        path_and_name: Optional[str] = None,
        object_ref: Optional[Any] = None,
        return_type_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
        )

        self.is_static = False

    def __call__(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Optional[Union[Any, CallableT]]:
        if hasattr(self, "client") and self.client is not None:
            return_tensor_type_pointer_type = self.client.lib_ast.query(
                path=self.return_type_name
            ).pointer_type

            ptr = return_tensor_type_pointer_type(client=self.client)
            if self.path_and_name is not None:
                msg = GetOrSetPropertyAction(
                    path=self.path_and_name,
                    id_at_location=ptr.id_at_location,
                    address=self.client.address,
                )
                self.client.send_immediate_msg_without_reply(msg=msg)
            return ptr

        path = kwargs["path"]
        index = kwargs["index"]

        if len(path) == index:
            return self.object_ref
        else:
            return self.attrs[path[index]](path=path, index=index + 1)
