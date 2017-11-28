#!/usr/bin/python3
import pandas as pd
import numpy as np
import sys

pd_data = pd.read_csv(sys.argv[1],header=None)
np_data = np.array(pd_data)
np.save(sys.argv[1] + ".npy",np_data)
