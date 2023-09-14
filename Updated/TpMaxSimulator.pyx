#!python
# cython: embedsignature=True, binding=True

import numpy as np
cimport numpy as np 
import cython

ctypedef np.double_t DTYPE_t
@cython.boundscheck(False)
def TpMaxSimulator(int _njobs, int _burnin, 
                                                               np.ndarray[DTYPE_t, ndim=1] sTime1,
                                                               np.ndarray[DTYPE_t, ndim=1] sTime2,
                                                               np.ndarray[DTYPE_t, ndim=1] sTime3,
                                                               int b1, int b2):

    cdef int i
    cdef double t

    cdef np.ndarray[np.double_t,
                ndim=2] ExitTimes = \
                np.zeros((3, _njobs+1))
    
    for i in range(1, _njobs+1):
            t = sTime1[i-1];
            if (ExitTimes[1,max(0,i-b1)] <= ExitTimes[0,i-1]+t):
                ExitTimes[0,i] = ExitTimes[0,i-1]+t;
            else:
                ExitTimes[0,i] = ExitTimes[1,max(0,i-b1)];
            
            t = sTime2[i-1];
            if(ExitTimes[1,i-1]>ExitTimes[0,i]):
                if(ExitTimes[2,max(0,i-b2)] <= ExitTimes[1,i-1]+t):
                    ExitTimes[1,i] = ExitTimes[1,i-1]+t;
                else:
                    ExitTimes[1,i] = ExitTimes[2,max(0,i-b2)];
            else:
                if (ExitTimes[2,max(0,i-b2)] <= ExitTimes[0,i]+t):
                    ExitTimes[1,i] = ExitTimes[0,i]+t;
                else:
                    ExitTimes[1,i] = ExitTimes[2,max(0,i-b2)];

            t=sTime3[i-1];
            if (ExitTimes[2,i-1] <= ExitTimes[1,i]):
                ExitTimes[2,i] = ExitTimes[1,i]+t;
            else:
                ExitTimes[2,i] = ExitTimes[2,i-1]+t;
                
    tp = (_njobs-_burnin)/(ExitTimes[2,_njobs]-ExitTimes[2,_burnin]);
    return tp