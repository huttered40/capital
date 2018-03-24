#!/usr/bin/env python

import numpy as np

#x = 1.48234
#y = 1.48235
x = 1.482341231423432344234234
y = 1.482341231423555555555

x_dbl = np.float64(x)
y_dbl = np.float64(y)
diff_dbl = x_dbl - y_dbl
#diff_true = x-y
diff_true = -1.232300000000000000000e-13

print(repr(x_dbl))
print(repr(y_dbl))
print(repr(diff_dbl))
print(diff_true)
rel_error = np.abs(diff_dbl - diff_true)/diff_true
print(repr(rel_error))
