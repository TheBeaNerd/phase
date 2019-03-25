#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv

x = np.array([[1,1],[1,-1]])
y = np.array([[0],[1]])
z = x.dot(y)
#print(z)

def sqpulse(time=0,delay=0,lnperiod=0):
    assert(lnperiod >= 0)
    if (lnperiod == 0):
        return 1
    radius = 2 ** (lnperiod - 1)
    q = (time + delay) // radius
    return 2 * (1 - (q % 2)) - 1

def sqwave(delay=0,lnperiod=0,lnframe=0):
    return [sqpulse(time=time,delay=delay,lnperiod=lnperiod) for time in range(0,2 ** lnframe)]

def ppulse(time=0,delay=0,lnperiod=0):
    assert(lnperiod >= 0)
    if (lnperiod == 0):
        return 1
    radius = 2 ** (lnperiod - 1)
    q = (time + delay) // radius
    s = 2 * (1 - (q % 2)) - 1
    return s if (((time + delay) % radius) == 0) else 0

def pwave(delay=0,lnperiod=0,lnframe=0):
    return [ppulse(time=time,delay=delay,lnperiod=lnperiod) for time in range(0,2 ** lnframe)]

# print(sqwave(delay=0,lnperiod=0,lnframe=3))
# print(sqwave(delay=0,lnperiod=1,lnframe=3))
# print(sqwave(delay=0,lnperiod=2,lnframe=3))
# print(sqwave(delay=0,lnperiod=3,lnframe=3))

print(pwave(delay=0,lnperiod=0,lnframe=3))
print(pwave(delay=0,lnperiod=1,lnframe=3))
print(pwave(delay=0,lnperiod=2,lnframe=3))
print(pwave(delay=0,lnperiod=3,lnframe=3))

def IM(lnframe=0):
    """Identity Matrix"""
    size = 2 ** lnframe
    return [[(1 if (row == col) else 0) for row in range(0,size)] for col in range(0,size)]

# print(IM(lnframe=3))

def lnrange(lnperiod=0):
    if (lnperiod < 1):
        return range(0,1)
    return range(0,2 ** (lnperiod - 1))

def lnprange(lnframe=0):
    return (i for j in (range(1,lnframe+1), range(0,1)) for i in j)

def SM(lnframe=0):
    """Square Matrix"""
    if (lnframe <= 0):
        return [[1]]
    return [pwave(delay=delay,lnperiod=lnperiod,lnframe=lnframe)  for lnperiod in lnprange(lnframe) for delay in lnrange(lnperiod)]

def pad(m=[[]],pre=0,post=0):
    prepad = [0 for _ in range(0,pre)]
    postpad = [0 for _ in range(0,post)]
    return [prepad + row + postpad for row in m]

def stage(lnstage=0,lnframe=0):
    if (lnstage == 0):
        return SM(lnframe)
    if (lnstage >= lnframe):
        return IM(lnframe)
    pre  = 0
    mid  = 0
    res = []
    ##post = (2 ** lnframe) - (pre + mid)
    for lnperiod in range(0,lnframe):
        pre = pre + mid
        mid = 2 ** lnperiod
        post = (2 ** lnframe) - (pre + mid)
        res += pad(stage(lnstage-1,lnperiod),pre=pre,post=post)
    pre = pre + mid
    mid = 1
    post = (2 ** lnframe) - (pre + mid)
    res += pad([[1]],pre=pre,post=post)
    return res

def xform(lnframe=0):
    res = np.array(IM(lnframe))
    for st in range(0,lnframe):
        res = np.array(stage(st,lnframe)).dot(res)
    return np.array([row for row in reversed(res)])

def ln2(x):
    for n in range(0,x):
        if (2 ** n >= x):
            return n
    return 0

def spectrum(input):
    lnframe = ln2(len(input))
    assert(2 ** lnframe == len(input))
    vector = np.matrix(input).T
    for st in range(0,lnframe):
        M = np.matrix(stage(st,lnframe))
        vector = M * vector
    return vector[::-1]

# print("x")
# print(np.array(stage(0,4)))
# print(np.array(stage(1,4)))
# print(np.array(stage(2,4)))
# print(np.array(stage(3,4)))
# print("x")
# print(xform(4))
# print(inv(np.matrix(xform(4))) * 32)

L1 = 2*(np.array([1,0,0,1,1,0,1,0,1,1,1,1,0,0,0,0]) - 0.5)
L2 = 2*(np.array([0,1,0,0,1,1,0,1,0,1,1,1,1,0,0,0]) - 0.5)

L3 = np.concatenate((L2[0:-1],L2[0:1]))

#print("L2 = " + str(L2))
#print("L3 = " + str(L3))

print(spectrum(L1))
print(np.matrix(xform(4))*np.matrix(L1).T)

print(spectrum(L2))
print(np.matrix(xform(4))*np.matrix(L2).T)

#print(spectrum([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
#print(spectrum([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
print(xform(4))

def zfixv(v):
    return 0.0 if (abs(v) < 1e-6) else v

zfix = np.vectorize(zfixv)

def reciprocalv(v):
    return 0.0 if (v == 0.0) else 1/v

reciprocal = np.vectorize(reciprocalv)

def pinv(A):
    (u,s,vt) = np.linalg.svd(A, full_matrices=False)
    s = zfix(s)
    assert(np.allclose(A, np.dot(u, np.dot(np.diag(s), vt))))
    s = reciprocal(s)
    s = np.diag(s)
    st = np.transpose(s)
    v  = zfix(np.transpose(vt))
    ut = zfix(np.transpose(u))
    return (v,st,ut)


# |  0  1 |
# | -1  0 |

#     res = []
#     pre  = []
#     post = []
#     for lnperiod in range(0,lnframe):
#         if ():
#             for delay in lnrange(lnperiod):
#                 res += pad(IM(size),pre,post)
#                 res += pad(SM(size),pre,post)
                
# print(np.array(SM(0)))
# print(np.array(SM(1)))
# print(np.array(SM(2)))
# print(np.array(SM(3)))
# print(np.array(SM(4)))

T0 = np.matrix([[-1]])

T1 = np.matrix([[0,-1],[1,0]])

I2 = np.matrix([[  4,   2,   0,  -2 ],
                [  0,   0,   4,   4 ],
                [  0,   4,   4,   0 ],
                [  0,   2,   0,   2 ]])

N2 = np.matrix([[ 2,   0,  -2,  -4 ],
                [ 0,   4,   4,   0 ],
                [ 4,   4,   0,   0 ],
                [ 2,   0,   2,   0 ]])

T2 = N2*inv(I2)

# (U,S,V) = pinv(T2)

# print(U)
# print(S)
# print(V)

# print(zfix(V * U))
