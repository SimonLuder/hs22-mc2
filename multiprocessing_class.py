import numpy as np
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager


        
class ReconstructSVDMultiprocessing():
    
    def __init__(self, hsplit=2, wsplit=2):
        self.colors = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m', '\033[98m']
        self.hsplit = hsplit
        self.wsplit = wsplit
        
        
    def get_args(self, ):
        
        args = []
        u_ = np.array_split(self.u, self.hsplit, axis=0)
        vt_ = np.array_split(self.vt.T, self.wsplit, axis=1)
        
        u_from, u_to,  = 0, 0
        for h in range(self.hsplit):
            u_to += u_[h].shape[0]
            
            vt_from, vt_to = 0, 0
            for w in range(self.wsplit):
                vt_to += vt_[w].shape[1]
                
                args.append(((u_from, u_to), (vt_from, vt_to), self.k, h*self.wsplit+w))
                
                vt_from += vt_[w].shape[1]
            u_from += u_[h].shape[0]
        return args

  
    def start_processes(self, u,s,vt,k):
        
        self.u, self.s, self.vt, self.k = u,s,vt,k
        args = self.get_args()
        
        with SharedMemoryManager() as smm:
            self.shm = smm.ShareableList([float(0)]*(u.shape[0]*vt.shape[1]))
            
            process = []
            for i in range(self.hsplit * self.wsplit):
                p = Process(target=self.reconstruct_svd_for_loops2, args=args[i])
                p.start()
                process.append(p)

            for p in process:
                p.join()

    def reconstruct_svd_for_loops2(self, u_idx, vt_idx, k, c):
        for i in range(u_idx[0],u_idx[1]):
            for j in range(vt_idx[0],vt_idx[1]):
                self.shm[i*self.vt.shape[0]+j] = float(np.dot(self.u[i,:k].reshape(1,-1), np.dot(np.diag(self.s)[:k, :k], self.vt[:k, j].reshape(-1,1)))[0, 0])
        
    def get_reconstruction(self):
        return np.array(self.shm).reshape((self.u.shape[0], self.vt.shape[1]))
  
