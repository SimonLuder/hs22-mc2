import os
import math
import numpy as np
from numba import vectorize
from numba import cuda, float32

im = np.random.normal(size=[3000, 2000])
u,s,vt = np.linalg.svd(im, full_matrices=False)

u = u.astype("float32")
s = s.astype("float32")
vt = vt.astype("float32")


@cuda.jit
def reconstruct_svd_cuda_2(u, s, vt, reco):
    """
    Perform matrix multiplication of reco = u * s * vt using CUDA shared memory.
    """
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


def reconstruct_svd_numba_setup(u, s, vt, k, reconstruction_method, TPB=32, verbose=False):

    reco = np.zeros((u.shape[0], vt.shape[1])).astype("float32")

    # to_device funktioniert nur wenn nochmals np.array ums slicing herum. why???
    u = np.array(u[:, 0:k])
    s = np.array(s[0:k])
    vt = np.array(vt[0:k, :])

    #copy to GPU
    u_gpu = cuda.to_device(u)
    s_gpu = cuda.to_device(s)
    vt_gpu = cuda.to_device(vt)
    reco_gpu = cuda.to_device(reco)

    threadsperblock = (TPB, TPB)
    grid_y_max = max(u.shape[0], vt.shape[0])
    grid_x_max = max(u.shape[1], vt.shape[1])
    blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
    blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if verbose:
        print(f"Blocks Per Grid: {blockspergrid}\nThreads Per Block: {threadsperblock}\n")

    reconstruction_method[blockspergrid, threadsperblock](u_gpu, s_gpu, vt_gpu, reco_gpu)

    # copy to CPU
    reco = reco_gpu.copy_to_host()
    return reco


def reconstruct_svd_numba_cuda(m_numba, m_cuda, TPB):
    return lambda u, s, vt, k: m_numba(u, s, vt, k, m_cuda, TPB)


if __name__ == "__main__":
    TPB=32
    k=min(im.shape)
    reco_svd_cuda_1 = reconstruct_svd_numba_cuda(reconstruct_svd_numba_setup, reconstruct_svd_cuda_2, TPB)
    print(reco_svd_cuda_1(u, s, vt, k))
