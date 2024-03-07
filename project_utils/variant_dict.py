import numpy as np 

VARIANT_DICT = {"ACCESS-ESM1-5": np.array(["r"+str(i)+"i1p1f1" for i in range(1,31)]), 
                "CNRM-CM6-1": np.array(["r"+str(i)+"i1p1f2" for i in range(1,31)]), 
                "CanESM5": np.array(["r"+str(i)+"i1p2f1" for i in range(1,31)]), 
                "GISS-E2-1-G": np.array(["r"+str(i)+"i1p1f1" for i in range(1,11)] + \
                                        ["r"+str(i)+"i1p3f1" for i in range(1,7)] + ["r"+str(i)+"i1p3f1" for i in range(8,11)] + \
                                        ["r"+str(i)+"i1p5f1" for i in range(1,5)] + ["r101i1p1f1"] + \
                                        ["r102i1p1f1"] + ["r"+str(i)+"i1p1f2" for i in range(1,6)]), 
                "IPSL-CM6A-LR": np.array(["r"+str(i)+"i1p1f1" for i in range(1,31)]), 
                "MIROC-ES2L": np.array(["r"+str(i)+"i1p1f2" for i in range(1,31)]), 
                "MIROC6": np.array(["r"+str(i)+"i1p1f1" for i in range(1,31)]), 
                "MPI-ESM1-2-LR": np.array(["r"+str(i)+"i1p1f1" for i in range(1, 31)]), 
                "NorCPM1": np.array(["r"+str(i)+"i1p1f1" for i in range(1, 31)])}


train_index = np.array([10, 4, 15, 6, 18, 9, 7, 22, 25, 26, 24, 5, 23, 12, 21, 8, 14, 19, 29, 11, 16, 3])
val_index = np.array([2, 13, 20])
test_index = np.array([27, 0, 28, 17, 1])