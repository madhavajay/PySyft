# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Union

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft

# syft relative
from ...core.common import UID
from ...core.common.serde.serializable import Serializable
from ...core.common.serde.serializable import bind_protobuf
from ...core.common.serde.serialize import _serialize
from ...core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr

module_type = type(syft)

# this will overwrite the .serializable_wrapper_type with an auto generated
# wrapper which will basically just hold the object being wrapped.
def GenerateWrapper(
    wrapped_type: type,
    import_path: str,
    protobuf_scheme: GeneratedProtocolMessageType,
    type_object2proto: CallableT,
    type_proto2object: CallableT,
    cons_new_obj: Optional[CallableT] = None
) -> None:
    @bind_protobuf
    class Wrapper(Serializable):
        def __init__(self, value: object):
            self.obj = value

        def _object2proto(self) -> Any:
            return type_object2proto(self.obj)

        @staticmethod
        def _proto2object(proto: Any) -> Any:
            obj = type_proto2object(proto)
            return Wrapper(value=obj)

        @staticmethod
        def get_protobuf_schema() -> GeneratedProtocolMessageType:
            return protobuf_scheme

        @staticmethod
        def construct_new_object(
            id: UID,
            data: StorableObject,
            description: Optional[str],
            tags: Optional[List[str]],
        ) -> StorableObject:
            if cons_new_obj:
                return cons_new_obj(
                    id=id, 
                    data=data, 
                    description=description, 
                    tags=tags
                )
            else:
                setattr(data, "_id", id)
                data.tags = tags
                data.description = description
                return data

        def upcast(self) -> Any:
            return self.obj

    # set __module__ and __name__
    module_parts = import_path.split(".")
    klass = module_parts.pop()
    Wrapper.__name__ = f"{klass}Wrapper"
    Wrapper.__module__ = f"syft.wrappers.{'.'.join(module_parts)}"
    # create a fake module `wrappers` under `syft`
    if not "wrappers" in syft.__dict__:
        syft.__dict__["wrappers"] = module_type(name="wrappers")
    # for each part of the path, create a fake module and add it to it's parent
    parent = syft.__dict__["wrappers"]
    for n in module_parts:
        if not n in parent.__dict__:
            parent.__dict__[n] = module_type(name=n)
        parent = parent.__dict__[n]
    # finally add our wrapper class to the end of the path
    parent.__dict__[Wrapper.__name__] = Wrapper

    aggressive_set_attr(
        obj=wrapped_type, name="serializable_wrapper_type", attr=Wrapper
    )

    def serialize(  # type: ignore
        self,
        to_proto: bool = True,
        to_bytes: bool = False,
    ) -> Union[str, bytes, Message]:
        return _serialize(
            obj=self,
            to_proto=to_proto,
            to_bytes=to_bytes,
        )

    serialize_attr = "serialize"
    if not hasattr(wrapped_type, serialize_attr):
        aggressive_set_attr(obj=wrapped_type, name=serialize_attr, attr=serialize)
    else:
        serialize_attr = "sy_serialize"
        aggressive_set_attr(obj=wrapped_type, name=serialize_attr, attr=serialize)
