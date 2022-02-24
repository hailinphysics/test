import numpy as np
from multiprocessing import shared_memory, Process
from multiprocessing import current_process
import psutil
from tqdm import tqdm


cpu_count = psutil.cpu_count(logical=False)
if cpu_count > 3:
    cpu_count -= 2


# 分配任务
def task_allocation(row_num, cpu_count):
    rand_idx = np.array(list(range(row_num)))
    np.random.shuffle(rand_idx)

    per_cpu = int(row_num / cpu_count)
    row_dict = {}
    for i in range(cpu_count - 1):
        idx_list = list(range(i * per_cpu, i * per_cpu + per_cpu))
        row_dict[i] = [rand_idx[idx] for idx in idx_list]

    idx_list = list(range(idx_list[-1] + 1, row_num - 1))
    row_dict[cpu_count - 1] = [rand_idx[idx] for idx in idx_list]

    return row_dict


# 把numpy数组放入共享内存
def create_nparray_shm(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)

    # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)

    # Copy the original data into shared memory
    np_array[:] = arr[:]
    return np_array, shm, arr.shape, arr.dtype


# 从共享内存中恢复numpy数组，进行处理
def process_func(ashm, ashape, atype, bshm, bshape, btype, row_list):
    a = np.ndarray(ashape, dtype=atype, buffer=ashm.buf)
    b = np.ndarray(bshape, dtype=btype, buffer=bshm.buf)

    for i in tqdm(row_list):
        for j in range(i+1, a.shape[0]):
            b[i, j] = a[i, j]


if current_process().name == "MainProcess":
    a = np.ones((1000, 1000))
    b = np.zeros((a.shape[0], a.shape[0]))

    shaper = a.shape
    shapew = b.shape

    a, ashm, ashape, atype = create_nparray_shm(a)
    b, bshm, bshape, btype, = create_nparray_shm(b)

    print('a', a)
    print(ashm)

    print('b', b)
    print(bshm)

    row_dict = task_allocation(b.shape[0], cpu_count)

    process_list = []
    for i in range(cpu_count):
        process = Process(target=process_func, args=(ashm, ashape, atype, bshm, bshape, btype, row_dict[i], ))
        process_list.append(process)

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    print('a', a)
    print('b', b)

    ashm.close()
    ashm.unlink()

    bshm.close()
    bshm.unlink()


