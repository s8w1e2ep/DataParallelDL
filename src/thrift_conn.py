import thriftpy
from thriftpy.rpc import make_client
from thriftpy.rpc import make_server
import os.path

def _get_thrift_file():
    import tempfile
    import requests
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".thrift")
    ret = requests.get("https://raw.githubusercontent.com/CheHsuan/DataParallelDL/master/src/weightsync.thrift")
    f.write(ret.content)
    f.flush()
    return f

def init_server(ip, port, requestHandler):
    if not os.path.exists("weightsync.thrift"):
        f = _get_thrift_file()
        weightsync_thrift = thriftpy.load(f.name, module_name="weightsync_thrift")
        f.close()
    else:
        weightsync_thrift = thriftpy.load("weightsync.thrift", module_name="weightsync_thrift")
    server = make_server(weightsync_thrift.WeightSync, requestHandler, ip, port)
    return server

def init_conn(ip, port):
    if not os.path.exists("weightsync.thrift"):
        f = _get_thrift_file()
        weightsync_thrift = thriftpy.load(f.name, module_name="weightsync_thrift")
        f.close()
    else:
        weightsync_thrift = thriftpy.load("weightsync.thrift", module_name="weightsync_thrift")
    client = make_client(weightsync_thrift.WeightSync, ip, port)
    return client
