#!/usr/bin/env python

import numpy as np
#import sys
#sys.path.append('../../../../ExternalLibraries/CTF/ctf/lib_python/')
import ctf

n=4
A = np.random.random((n,n))
B = np.random.random((n,n))
tA = ctf.astensor(A)
tB = ctf.astensor(B)
tC = ctf.dot(tA,tB)
print("This is tensor C - ", tC)