import torch
import numpy as np
import os
import time
import shutil
import subprocess

import parameters
import shlex


def compute_rFEM(file_del, steps_counter, worker_ID):
    #worker_ID = int(worker_ID)
    worker_ID = str(worker_ID)
    cmd = 'docker exec -w /mnt/io/build/drivers/beamHomogenization/ ae108-legacy-dev-1 ' \
          './BeamHomogenization-fromFile2D --name=' + worker_ID

    # check if tangent matrix exists, delete it every xy iterations to keep read in time io
    path_rel_tangent = parameters.path_laptop + '/beamHomogenization/tangent_matrix' + worker_ID
    file_rem_time = time.time()
    if os.path.isfile(path_rel_tangent):
        pass
        #os.remove(path_rel_tangent)
        #print("tangent removed")
    fp = open(parameters.path_laptop + '/beamHomogenization/tangent_matrix' + worker_ID, 'w')
    fp.close()

    filesize_old = os.path.getsize(parameters.path_laptop + '/beamHomogenization/tangent_matrix'
                                   '' + worker_ID)


    # run FEM
    file_rem_time_stop = time.time()
    if parameters.time_measure: print("file removal time", file_rem_time_stop - file_rem_time)
    run_time = time.time()
    subprocess.Popen(shlex.split(cmd))
    run_time_end = time.time()
    if parameters.time_measure: print("runtime", run_time_end-run_time)
    time.sleep(0.05)
    run_check_time = time.time()
    filesize = os.path.getsize(parameters.path_laptop + '/beamHomogenization/tangent_matrix' +
                               worker_ID)

    for i in range(10):
        if (filesize - filesize_old) > 0.0:
            FEM_failed = 0
            stiffness_real = torch.from_numpy(np.genfromtxt(path_rel_tangent, delimiter=',', dtype=np.float32)[-1, 2:],)
            run_check_time_end = time.time()
            if parameters.time_measure: print("read in time", run_check_time_end-run_check_time)
            return stiffness_real, FEM_failed
        else:
            time.sleep(0.25)
            filesize = os.path.getsize(
                parameters.path_laptop + '/beamHomogenization/tangent_matrix' + worker_ID)
    print("FEM failed")
    FEM_failed = 1
    return torch.rand(9), FEM_failed