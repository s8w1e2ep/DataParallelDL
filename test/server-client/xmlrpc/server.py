import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer

def upload(model):
    return model

server = SimpleXMLRPCServer(("localhost", 8000))
print "Listening on port 8000..."
server.register_function(upload, "upload")
server.serve_forever()
