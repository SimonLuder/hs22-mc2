import multiprocessing as mp
from numba import vectorize, cuda, float32
import numpy as np
import imageio
from PIL import Image
import math
import sys

def reconstruct_svd_cuda_mp(files, num_processes):

    # parent process starts a fresh Python interpreter process
    ctx = mp.get_context('spawn')
    manager = mp.Manager()
    file_queue = manager.Queue()
    reco_queue = manager.Queue()

    # put file names in queue
    for file in files:
        file_queue.put(file)

    for n in range(num_processes):
        file_queue.put(None)

    processes = []
    for _ in range(num_processes):
        p = ctx.Process(target=reconstruct_svd_numba_setup_mp, args=[file_queue, reco_queue])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return reco_queue



def reconstruct_svd_numba_setup_mp(file_queue, reco_queue, TPB=32, verbose=False):

    # setup separate cuda stream for this process
    stream = cuda.stream()

    while True:

        file = file_queue.get()

        if file is None:
            break

        (u, s, vt, k) = file

        # im = Image.open(file).convert('L')
        # im = im - np.min(im) / np.max(im) - np.min(im) # normalize data
        # u, s, vt = np.linalg.svd(im, full_matrices=False)

        # if k is None:
        #     # k = np.min(im.shape)
        #     k = len(s)

        u = np.array(u[:, 0:k])
        s = np.array(s[0:k])
        vt = np.array(vt[0:k, :])
        reco = np.zeros((u.shape[0], vt.shape[1])).astype("float32")


        threadsperblock = (TPB, TPB)
        grid_y_max = max(u.shape[0], vt.shape[0])
        grid_x_max = max(u.shape[1], vt.shape[1])
        blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
        blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        with cuda.pinned(u, s, vt, reco):
            #copy to GPU on process stream
            u_gpu = cuda.to_device(u, stream=stream)
            s_gpu = cuda.to_device(s, stream=stream)
            vt_gpu = cuda.to_device(vt, stream=stream)
            reco_gpu = cuda.to_device(reco, stream=stream)

            # # reco_queue.put((file, k))
            reconstruct_svd_cuda_2[blockspergrid, threadsperblock, stream](u_gpu, s_gpu, vt_gpu, reco_gpu)

            # copy to CPU on process stream
            reco = reco_gpu.copy_to_host(stream=stream)

            reco_queue.put((reco))

    sys.stdout.flush()


@cuda.jit
def reconstruct_svd_cuda_3(u, s, vt, reco):
    """
    Perform matrix multiplication of reco = u * s * vt using CUDA shared memory.
    """

    # TPB = cuda.blockDim.x
    TPB = 32

    # Define an array in the shared memory
    u_shared1 = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    s_shared1 = cuda.shared.array(shape=(TPB), dtype=float32)
    vt_shared1 = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    u_shared2 = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    s_shared2 = cuda.shared.array(shape=(TPB), dtype=float32)
    vt_shared2 = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2) # 2 dim
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.y    # blocks per grid

    u_shared1[ty, tx] = 0
    if tx < 1:
        s_shared1[tx] = 0
    vt_shared1[ty, tx] = 0

    if y < u.shape[0] and tx < u.shape[1]:
        u_shared1[ty, tx] = u[y, tx]

    if tx < s.shape[0]:
        s_shared1[tx] = s[tx]

    if x < vt.shape[1] and ty < vt.shape[0]:
        vt_shared1[ty, tx] = vt[ty, x]

    cuda.syncthreads()

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(1, bpg + 1):

        # update shared memory
        u_shared1, u_shared2 = u_shared2, u_shared1
        s_shared1, s_shared2 = s_shared2, s_shared1
        vt_shared1, vt_shared2 = vt_shared2, vt_shared1

        u_shared1[ty, tx] = 0
        if tx < 1:
            s_shared1[tx] = 0
        vt_shared1[ty, tx] = 0

        if y < u.shape[0] and (tx + i * TPB) < u.shape[1]:
            u_shared1[ty, tx] = u[y, tx + i * TPB]

        if (tx + i * TPB) < s.shape[0]:
            s_shared1[tx] = s[tx + i * TPB]

        if x < vt.shape[1] and (ty + i * TPB) < vt.shape[0]:
            vt_shared1[ty, tx] = vt[ty + i * TPB, x]

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += u_shared2[ty, j] * s_shared2[j] * vt_shared2[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()


    if y < reco.shape[0] and x < reco.shape[1]:
        reco[y, x] = tmp


@cuda.jit
def reconstruct_svd_cuda_2(u, s, vt, reco):
    """
    Perform matrix multiplication of reco = u * s * vt using CUDA shared memory.
    """
    TPB = 32
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    u_shared = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    s_shared = cuda.shared.array(shape=(TPB), dtype=float32)
    vt_shared = cuda.shared.array(shape=(TPB, TPB), dtype=float32)


    x, y = cuda.grid(2) # 2 dim
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.y    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        u_shared[ty, tx] = 0
        if tx < 1:
            s_shared[tx] = 0
        vt_shared[ty, tx] = 0

        if y < u.shape[0] and (tx + i * TPB) < u.shape[1]:
            u_shared[ty, tx] = u[y, tx + i * TPB]

        if (tx + i * TPB) < s.shape[0]:
            s_shared[tx] = s[tx + i * TPB]

        if x < vt.shape[1] and (ty + i * TPB) < vt.shape[0]:
            vt_shared[ty, tx] = vt[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += u_shared[ty, j] * s_shared[j] * vt_shared[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()

    if y < reco.shape[0] and x < reco.shape[1]:
        reco[y, x] = tmp



if __name__ == "__main__":

    im = np.random.normal(size=[3000, 2000])
    im = im - np.min(im) / np.max(im) - np.min(im)

    k=min(im.shape)
    u, s, vt = np.linalg.svd(im, full_matrices=False)
    deco_imgs = [(u, s, vt, k)]*20

    reconstruct_svd_cuda_mp(deco_imgs, 5)
