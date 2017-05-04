from thrift_conn import init_conn
import time

conn = init_conn("127.0.0.1", 50001)

for i in range(10):
    print i
    conn.upload("m"+str(i))
    time.sleep(1)    
