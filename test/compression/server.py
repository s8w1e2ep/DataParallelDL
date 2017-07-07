import thriftpy
test_thrift = thriftpy.load("benchmarkmessage.thrift", module_name="test_thrift")

from thriftpy.rpc import make_server

class Dispatcher(object):
    def echo(self, message):
        return message

server = make_server(test_thrift.Helloworld, Dispatcher(), '127.0.0.1', 6000)
server.serve()
