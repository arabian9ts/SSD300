"""
compute required indicies.

date: 10/1
author: arabian9ts
"""

def smooth_L1(x):
    y = 0
    if abs(x) < 1:
        y = 0.5 * x**2
    else:
        y = abs(x) - 0.5

    return y

print(smooth_L1(1))
print(smooth_L1(0.1))