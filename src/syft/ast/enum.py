# stdlib
from typing import Any
from typing import Optional

# syft relative
from .. import ast
from ..core.node.common.action.get_enum_attribute_action import EnumAttributeAction


class EnumAttribute(ast.attribute.Attribute):
    def __init__(
        self,
        parent: ast.attribute.Attribute,
        path_and_name: Optional[str] = None,
        return_type_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.parent = parent
        super().__init__(
            path_and_name=path_and_name,
            return_type_name=return_type_name,
            client=client,
        )

    def get_remote_enum_attribute(self):
        if self.path_and_name is None:
            raise ValueError("MAKE PROPER SCHEMA - Can't get enum attribute")

        return_tensor_type_pointer_type = self.client.lib_ast.query(
            path=self.return_type_name
        ).pointer_type

        ptr = return_tensor_type_pointer_type(client=self.client)

        msg = EnumAttributeAction(
            path=self.path_and_name,
            id_at_location=ptr.id_at_location,
            address=self.client.address,
        )
        self.client.send_immediate_msg_without_reply(msg=msg)
        return ptr

    def solve_get_enum_attribute(self) -> int:
        return getattr(self.parent.object_ref, self.path_and_name.rsplit(".")[-1])

    def __call__(self, *args, **kwargs) -> None:
        raise ValueError("MAKE PROPER SCHEMA, THIS SHOULD NEVER BE CALLED")

    def add_path(self, *args, **kwargs) -> None:
        raise ValueError("MAKE PROPER SCHEMA")

