import multiprocessing as mp
from sympy import Matrix
import numpy as np
import h5py
import sys

num_processes = mp.cpu_count()
num_processes = 24

# Calculate null-space for all training points
def get_nullspace(J):
    A = Matrix(np.transpose(J))
    n = np.array(A.nullspace()).astype(np.float32)

    # make unit-vectors
    n = n*1.0/np.sqrt(np.sum(np.square(n), axis=-1)[:, None])
    return n

# Load Jacobians
J_train = np.load('J_train_mnist.npy')
J_test = np.load('J_test_mnist.npy')

print(J_train.shape)
print(J_test.shape)

n_train_shape = (J_train.shape[0], J_train.shape[1]-J_train.shape[2], J_train.shape[1])
n_test_shape = (J_test.shape[0], J_test.shape[1]-J_test.shape[2], J_test.shape[1])



def write_h5py(queue):
    hdf5_file_name = '/data/sverneka/full_mnist_0_9_cond_latent_dim_8/nullspaces_mnist.hdf5'
    hdf5_file = h5py.File(hdf5_file_name, mode='w')

    hdf5_file.create_dataset("n_train", n_train_shape, dtype='float32')
    hdf5_file.create_dataset("n_test", n_test_shape, dtype='float32')

    while True:
	if not queue.empty():
            args = queue.get()
            if args:
		data_type, i, N = args
            	hdf5_file[data_type][i,:,:] = N
                if i%100==0:
		    print("wrote up to ", i)
                    sys.stdout.flush()
            else:
                break
    hdf5_file.close()


def create_nullspace(queue, write_queue):
    while not queue.empty():
        args = queue.get()
        data_type, i, J = args
        N = get_nullspace(J)
        write_queue.put([data_type, i, N])
        if i % 100 == 0:
            print("finished computing up to ", i)
            sys.stdout.flush()

def create_nullspace_main():
    output = mp.Queue()
    inqueue = mp.Queue()
    jobs = []
    proc = mp.Process(target=write_h5py, args=(output, ))
    proc.start()

    data_type = 'n_train'
    for i in range(J_train.shape[0]):
        inqueue.put([data_type, i, J_train[i]])

    data_type = 'n_test'
    for i in range(J_test.shape[0]):
        inqueue.put([data_type, i, J_test[i]])

    for i in range(num_processes):
        p = mp.Process(target=create_nullspace, args=(inqueue, output))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    output.put(None)
    proc.join()


create_nullspace_main()


