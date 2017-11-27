import socket
import sys
import numpy as np
import xgboost as xgb

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_addr = ("192.168.0.133",9999)
sock.bind(server_addr)
model = xgb.Booster()
model.load_model("/home/qinjianbo/modelXGB")

while True:
    data,address = sock.recvfrom(4096)
    data  = data.decode('utf-8').replace("[","").replace("]","")
    data  = np.fromstring(data,dtype=float,sep=",").reshape(1,39)
    featrue = xgb.DMatrix(data)
    print(model.predict(featrue))
