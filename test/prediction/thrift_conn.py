import thriftpy
from thriftpy.rpc import make_client
from thriftpy.rpc import make_server
import os.path
from predictor import predictor

def _get_thrift_file():
    import tempfile
    import requests
    print "Try to download thrift file..."
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".thrift")
    ret = requests.get("https://raw.githubusercontent.com/CheHsuan/DataParallelDL/master/test/prediction/ping.thrift")
    f.write(ret.content)
    f.flush()
    return f

def init_server(ip, port, requestHandler):
    if not os.path.exists("ping.thrift"):
        f = _get_thrift_file()
        ping_thrift = thriftpy.load(f.name, module_name="ping_thrift")
        f.close()
    else:
        ping_thrift = thriftpy.load("ping.thrift", module_name="ping_thrift")
    print requestHandler.ping()
    server = make_server(ping_thrift.PingService, requestHandler, ip, port)
    return server

def init_conn(ip, port):
    if not os.path.exists("ping.thrift"):
        f = _get_thrift_file()
        ping_thrift = thriftpy.load(f.name, module_name="ping_thrift")
        f.close()
    else:
        ping_thrift = thriftpy.load("ping.thrift", module_name="ping_thrift")
    conn = make_client(ping_thrift.PingService, ip, port)
    return conn
