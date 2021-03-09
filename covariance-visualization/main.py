import numpy as np
import math
import matplotlib.pyplot as plt

npoints = 1000
mean = [0, 0]
s1 = 9
s2 = 1
cov = [[s1, 0], [0, s2]]
# x^2 / 1 + y^2 / 100 = 5.99
# x^2 / (s1 * 5.99) + y^2 / (s2 * 5.99) = 1
# x = cos(t) / sqrt(s1 * 5.99)
# y = sin(t) / sqrt(s2 * 5.99)

x_o, y_o = np.random.multivariate_normal(mean, cov, npoints).T
#plt.plot(x_o, y_o, 'b.')
#plt.plot([math.cos(t) * math.sqrt(s1*5.99) for t in np.linspace(0, 2*math.pi, 100)],
#         [math.sin(t) * math.sqrt(s2*5.99) for t in np.linspace(0, 2*math.pi, 100)],
#         'g-')
#plt.axis('equal')

c = math.cos(math.pi / 4)
s = math.sin(math.pi / 4)
# [c  -s]
# [s   c]
print('Rotation matrix that transform points in O to P')
T_po = np.array([[c, -s],[s,c]])
print(T_po)

x_p = [e[0]*c-e[1]*s for e in zip(x_o, y_o)]
y_p = [e[0]*s+e[1]*c for e in zip(x_o, y_o)]
plt.plot(x_p, y_p, 'b.', label='Data points')
plt.axis('equal')

M = np.zeros((2,2))
for i, j in zip(x_p, y_p):
    M[0, 0] += i*i
    M[0, 1] += i*j
    M[1, 0] += j*i
    M[1, 1] += j*j

S, Q = np.linalg.eig(M / npoints)
print(S)
print('Rotation matrix that transform points in P to O')
print(Q)
ellipse_x_p = [math.cos(t) * math.sqrt(S[0]*5.99) for t in np.linspace(0, 2*math.pi, 100)]
ellipse_y_p = [math.sin(t) * math.sqrt(S[1]*5.99) for t in np.linspace(0, 2*math.pi, 100)]
ellipse_x_o = [e[0]*Q[0,0]+e[1]*Q[0,1] for e in zip(ellipse_x_p, ellipse_y_p)]
ellipse_y_o = [e[0]*Q[1,0]+e[1]*Q[1,1] for e in zip(ellipse_x_p, ellipse_y_p)]
plt.plot(ellipse_x_o, ellipse_y_o, 'g-', label='Covariance ellipse')

# A*q1
reduced_f_p = [e[0]*Q[0,0] + e[1]*Q[1,0] for e in zip(x_p, y_p)]
# We set other feature dimensions all be 0
fx_o = [e*Q[0,0]+0*Q[0,1] for e in reduced_f_p]
fy_o = [e*Q[1,0]+0*Q[1,1] for e in reduced_f_p]

plt.plot(fx_o, fy_o, 'rx', label='Reduced dimension features')
plt.legend()
plt.show()
