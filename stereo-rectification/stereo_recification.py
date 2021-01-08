#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt

def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

eps = 1E-8
def rectify(R12, t12_1, uv1_c, uv2_c, new_K1, new_K2):
    e1 = t12_1 / np.linalg.norm(t12_1)
    e2 = np.cross(np.array([0, 0, 1]), e1)
    # e2 = np.cross(e1, np.array([0, 0, 1]))
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    R_1_rect = np.stack([e1, e2, e3], axis=1)
    R_rect_1 = np.transpose(R_1_rect)
    uv1_rect_c = R_rect_1.dot(uv1_c)
    uv1_rect_c = uv1_rect_c / uv1_rect_c[2]
    uv1_rect = new_K1.dot(uv1_rect_c)
    uv1_rect = uv1_rect / uv1_rect[2]
    uv2_rect_c = R_rect_1.dot(R12).dot(uv2_c)
    uv2_rect_c = uv2_rect_c / uv2_rect_c[2]
    uv2_rect = new_K2.dot(uv2_rect_c)
    uv2_rect = uv2_rect / uv2_rect[2]
    return (R_1_rect, uv1_rect_c, uv2_rect_c, uv1_rect, uv2_rect)

def triangulate(T_1, T_2, uv1_rect, uv2_rect):
    r1 = uv1_rect[0]*T_1[2, :] - T_1[0, :]
    r2 = uv1_rect[1]*T_1[2, :]- T_1[1, :]
    r3 = uv2_rect[0]*T_2[2, :] - T_2[0, :]
    r4 = uv2_rect[1]*T_2[2, :]- T_2[1, :]
    A = np.stack((r1,r2,r3,r4), axis=0)
    [u, s, vt] = np.linalg.svd(A)
    p_3d = vt[3, :] / vt[3,3]
    return p_3d[0:3]

K1 = np.array([[98, 0, 199],
              [0, 98, 198],
              [0, 0, 1]])
K1inv = np.linalg.inv(K1)
K2 = np.array([[102, 0, 198],
              [0, 102, 199],
              [0, 0, 1]])

K2inv = np.linalg.inv(K2)

P0 = np.array([1, 1, 10])

t12_1 = np.array([0.5, 0.01, 0.01])
r = 1* math.pi / 180
p = 1 * math.pi / 180
y = 1 * math.pi / 180
c = np.cos([r, p, y])
s = np.sin([r, p, y])
R12 = np.array([[1, 0, 0],
                [0, c[0], -s[0]],
                [0, s[0], c[0]]]).dot(
                    np.array([[c[1], 0, s[1]],
                              [0,      1,  0],
                              [-s[1],    0,  c[1]]])).dot(
                                  np.array([[c[2], -s[2], 0],
                                            [s[2], c[2], 0],
                                            [0, 0, 1]]))

uv1 = K1.dot(P0)
normalized_uv1 = uv1 / uv1[2]
uv1_c = K1inv.dot(normalized_uv1)
uv1_c = uv1_c / uv1_c[2]

P1 = np.transpose(R12).dot(P0) - np.transpose(R12).dot(t12_1)
# assert((R12.dot(P1)+t12_1 == P0).all())

uv2 = K2.dot(P1)
normalized_uv2 = uv2 / uv2[2]
uv2_c = K2inv.dot(normalized_uv2)
uv2_c = uv2_c / uv2_c[2]

E = skew(t12_1).dot(R12)
epipolar_line_1 = E.dot(uv2_c)
assert(uv1_c.dot(epipolar_line_1) < eps)

T_1 = np.zeros((3,4))
T_1[0:3, 0:3] = K1.dot(np.eye(3))
T_1[0:3, 3] = K1.dot(np.array([0,0,0]))
T_2 = np.zeros((3,4))
T_2[0:3, 0:3] = K2.dot(np.transpose(R12))
T_2[0:3, 3] = K2.dot(-np.transpose(R12).dot(t12_1))
print(normalized_uv1, normalized_uv2)
P0_triangulated = triangulate(T_1, T_2, normalized_uv1, normalized_uv2)
print(P0, P0_triangulated)
print("=============")

# new K1 is not necessary same as K1
new_K1 = np.array([[100, 0, 180],
                   [0, 100, 180],
                   [0, 0, 1]])

# new K2 is not necessary same as K1, but you want Hx, Hy to be same as ones in new_K1
new_K2 = np.array([[100, 0, 180],
                   [0, 100, 180],
                   [0, 0, 1]])

(R_1_rect, uv1_rect_c, uv2_rect_c, uv1_rect, uv2_rect) = rectify(R12, t12_1, uv1_c, uv2_c, new_K1, new_K2)
T_1 = np.zeros((3,4))
T_1[0:3, 0:3] = new_K1.dot(np.transpose(R_1_rect))
T_1[0:3, 3] = new_K1.dot(np.array([0,0,0]))
T_2 = np.zeros((3,4))
T_2[0:3, 0:3] = new_K2.dot(np.transpose(R_1_rect))
T_2[0:3, 3] = new_K2.dot(-np.transpose(R_1_rect).dot(t12_1))
print(uv1_rect, uv2_rect)
P0_triangulated = triangulate(T_1, T_2, uv1_rect, uv2_rect)
print(P0, P0_triangulated)

uv1_xo = []
uv1_yo = []
uv2_xo = []
uv2_yo = []

uv1_x = []
uv1_y = []
uv2_x = []
uv2_y = []
for x in [0, 400]:
    for y in [0, 400]:
        uv1_c = K1inv.dot(np.array([x, y, 1]))
        uv1_c = uv1_c / uv1_c[2]
        uv2_c = K1inv.dot(np.array([x, y, 1]))
        uv2_c = uv2_c / uv2_c[2]
        (R_1_rect, uv1_rect_c, uv2_rect_c, uv1_rect, uv2_rect) = rectify(R12, t12_1, uv1_c, uv2_c, new_K1, new_K2)
        uv1_xo.append(x)
        uv1_yo.append(y)
        uv2_xo.append(x)
        uv2_yo.append(y)
        uv1_x.append(uv1_rect[0])
        uv1_y.append(uv1_rect[1])
        uv2_x.append(uv2_rect[0])
        uv2_y.append(uv2_rect[1])

plt.subplot(121)
plt.scatter(uv1_x, uv1_y, marker='o')
plt.scatter(uv1_xo, uv1_yo, marker='^', c='r')
plt.grid(True)
plt.subplot(122)
plt.scatter(uv2_x, uv2_y, marker='o')
plt.scatter(uv2_xo, uv2_yo, marker='^', c='r')
plt.grid(True)
plt.show()
