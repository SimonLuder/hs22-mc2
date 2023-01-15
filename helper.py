import timeit
import numpy as np

class TimeLogger():
    
    def __init__(self, u, s, vt):
        self.__logger = []
        self.u = u
        self.s = s
        self.vt = vt

    def log_time(self, method, k_range, repeat, number, verbose=False):
        if verbose:
            print(f"running with: {method.__name__}")
        
        for k in k_range:
            t = timeit.Timer(lambda: method(self.u, self.s, self.vt, k))
            time = t.repeat(repeat=repeat, number=number)

            mean, std = np.mean(time), np.std(time)
            
            log = dict()
            log["name"] = method.__name__
            log["k"] = k
            log["repeat"] = repeat
            log["number"] = number
            log["mean"] = mean
            log["std"] = std
                
            self.__logger.append(log)
                
    def get_log(self):
        return self.__logger