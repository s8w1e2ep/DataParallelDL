# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='service.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\rservice.proto\" \n\x0cmodelRequest\x12\x10\n\x08greeting\x18\x01 \x01(\t\"\x1e\n\rmodelResponse\x12\r\n\x05reply\x18\x01 \x01(\t2/\n\x04test\x12\'\n\x06upload\x12\r.modelRequest\x1a\x0e.modelResponseb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_MODELREQUEST = _descriptor.Descriptor(
  name='modelRequest',
  full_name='modelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='greeting', full_name='modelRequest.greeting', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=49,
)


_MODELRESPONSE = _descriptor.Descriptor(
  name='modelResponse',
  full_name='modelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reply', full_name='modelResponse.reply', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=81,
)

DESCRIPTOR.message_types_by_name['modelRequest'] = _MODELREQUEST
DESCRIPTOR.message_types_by_name['modelResponse'] = _MODELRESPONSE

modelRequest = _reflection.GeneratedProtocolMessageType('modelRequest', (_message.Message,), dict(
  DESCRIPTOR = _MODELREQUEST,
  __module__ = 'service_pb2'
  # @@protoc_insertion_point(class_scope:modelRequest)
  ))
_sym_db.RegisterMessage(modelRequest)

modelResponse = _reflection.GeneratedProtocolMessageType('modelResponse', (_message.Message,), dict(
  DESCRIPTOR = _MODELRESPONSE,
  __module__ = 'service_pb2'
  # @@protoc_insertion_point(class_scope:modelResponse)
  ))
_sym_db.RegisterMessage(modelResponse)


try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces


  class testStub(object):

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.upload = channel.unary_unary(
          '/test/upload',
          request_serializer=modelRequest.SerializeToString,
          response_deserializer=modelResponse.FromString,
          )


  class testServicer(object):

    def upload(self, request, context):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_testServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'upload': grpc.unary_unary_rpc_method_handler(
            servicer.upload,
            request_deserializer=modelRequest.FromString,
            response_serializer=modelResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'test', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetatestServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    def upload(self, request, context):
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetatestStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    def upload(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      raise NotImplementedError()
    upload.future = None


  def beta_create_test_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('test', 'upload'): modelRequest.FromString,
    }
    response_serializers = {
      ('test', 'upload'): modelResponse.SerializeToString,
    }
    method_implementations = {
      ('test', 'upload'): face_utilities.unary_unary_inline(servicer.upload),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_test_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('test', 'upload'): modelRequest.SerializeToString,
    }
    response_deserializers = {
      ('test', 'upload'): modelResponse.FromString,
    }
    cardinalities = {
      'upload': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'test', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)
