# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities

import service_pb2 as service__pb2


class testStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.upload = channel.unary_unary(
        '/test/upload',
        request_serializer=service__pb2.modelRequest.SerializeToString,
        response_deserializer=service__pb2.modelResponse.FromString,
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
          request_deserializer=service__pb2.modelRequest.FromString,
          response_serializer=service__pb2.modelResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'test', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
