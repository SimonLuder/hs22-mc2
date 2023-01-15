import os
import math
import numpy as np
from numba import vectorize
from numba import cuda, float32

im = np.random.normal(size=[3000, 3000])
u,s,vt = np.linalg.svd(im, full_matrices=False)

u = u.astype("float32")
s = s.astype("float32")
vt = vt.astype("float32")


@cuda.jit
def reconstruct_svd_cuda_1(u, s, vt, reco):

    j, i = cuda.grid(2)
    if i < reco.shape[0] and j < reco.shape[1]:
        tmp = 0.
        for n in range(s.shape[0]):
            tmp += u[i, n] * s[n] *  vt[n, j]
        reco[i, j] = tmp


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
    reco_svd_cuda_1 = reconstruct_svd_numba_cuda(reconstruct_svd_numba_setup, reconstruct_svd_cuda_1, TPB)
    print(reco_svd_cuda_1(u, s, vt, k))
