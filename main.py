import socket
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_addr = ("192.168.0.133",9999)
sock.bind(server_addr)
model = xgb.Booster()
model.load_model("/home/qinjianbo/modelXGB")

file = open(sys.argv[1],"w+")
flag = 1
while flag:
    data,address = sock.recvfrom(4096)
    data  = data.decode('utf-8').replace("[","").replace("]","")
    try:
        file.write(data)
        file.write("\n")
    except KeyboardInterrupt:
        file.close()
        pd_data = pd.read_csv(file,header=None)
        np_data = np.array(pd_data)
        np.save(sys.argv[1] + ".npy",np_data)
        flag = 0
#    data  = np.fromstring(data,dtype=float,sep=",").reshape(1,39)
#    featrue = xgb.DMatrix(data)
#    print(model.predict(featrue))

