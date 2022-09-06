# import math
# import random
import torch
# import ctypes
# import sys
#vimport pathlib
import numpy as np
import os
import time
import parameters
# import shutil
import subprocess
import shlex

"""
compute_rFEM runs the FEM code and returns the stiffness matrix as torch.tensor; mesh, tangent_matrix file and 
BeamHomogenization-fromFile2D must be saved at the pathes below
 """
def compute_rFEM(file_del, steps_counter, worker_ID):
    worker_ID = str(worker_ID)
    # check if BeamHomogenization-fromFile2D exists in the current directory:
    if not os.path.isfile("BeamHomogenization-fromFile2D"):
        print("BeamHomogenization-fromFile2D not found in current directory")
        # create a symbolic link to the BeamHomogenization-fromFile2D executable:
        os.symlink("/cluster/home/jaicher/MA/ae108-legacy/build/drivers/beamHomogenization/BeamHomogenization-fromFile2D", "BeamHomogenization-fromFile2D")
    if parameters.time_measure:
        path_rel = '../ae108-legacy/build/drivers/beamHomogenization/'

        cmd3 = './BeamHomogenization-fromFile2D --name=' + worker_ID

        # check if tangent matrix exists, delete it every xy iterations to keep read in time io
        path_rel_tangent = "tangent_matrix" + worker_ID
        file_rem_time = time.time()
        #if steps_counter % file_del == 0:
        try:
            os.remove(path_rel_tangent)
            fp = open('tangent_matrix' + worker_ID, 'w')
            fp.close()
        except:
            pass

        filesize_old = os.path.getsize("tangent_matrix" + worker_ID)
        file_rem_time_stop = time.time()
        print("file removal time", file_rem_time_stop - file_rem_time)
        # run FEM
        run_time = time.time()
        #os.system(cmd3)
        subprocess.Popen(shlex.split(cmd3))
        run_time_end = time.time()
        print("runtime", run_time_end-run_time)
        run_check_time = time.time()
        filesize = os.path.getsize("tangent_matrix" + worker_ID)

        for i in range(12):
            if (filesize - filesize_old) > 0.0:
                FEM_failed = 0
                stiffness_real = torch.from_numpy(np.genfromtxt(path_rel_tangent, delimiter=',', dtype=np.float32)[-1, 2:],)
                run_check_time_end = time.time()
                print("read in time", run_check_time_end-run_check_time)
                return stiffness_real, FEM_failed
            else:
                time.sleep(0.25)
                filesize = os.path.getsize("tangent_matrix" + worker_ID)
        print("FEM failed")
        FEM_failed = 1
        return torch.rand(9), FEM_failed

    else:
        cmd3 = './BeamHomogenization-fromFile2D --name=' + worker_ID

        path_rel_tangent = str("tangent_matrix" + worker_ID)
        try:
            #os.remove(path_rel_tangent)
            fp = open(path_rel_tangent, 'w')
            fp.close()
        except:
            pass

        filesize_old = os.path.getsize("tangent_matrix" + worker_ID)
        # run FEM
        subprocess.Popen(shlex.split(cmd3))
        time.sleep(0.05)

        # check, if FEM worked by checking if tangent matrix file is bigger than before
        filesize = os.path.getsize("tangent_matrix" + worker_ID)
        for i in range(30):
            if (filesize - filesize_old) > 0.0:
                FEM_failed = 0
                # read stiffness matrix
                stiffness_real = torch.from_numpy(np.genfromtxt(path_rel_tangent, delimiter=',', dtype=np.float32)[-1, 2:], )
                return stiffness_real, FEM_failed
            else:
                time.sleep(0.2)
                filesize = os.path.getsize("tangent_matrix" + worker_ID)
        print("FEM failed")
        FEM_failed = 1
        return torch.rand(9), FEM_failed
