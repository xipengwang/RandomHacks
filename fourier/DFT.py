from svgpathtools import svg2paths, parse_path
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math


def cal_coefficient(N, pts):
    # N >= 1:
    freq = list(range(0, N, 1))
    n = len(pts)
    coef = []
    for f in freq:
        c = complex(0, 0)
        for i, p in enumerate(pts):
            r, phase = cmath.polar(p)
            phase -=  2.0 * math.pi * f * i / n
            c += cmath.rect(r, phase)
        coef.append(c)
    return freq, coef


path_alt = parse_path('M 300 100 C 100 100 200 200 200 300 L 250 350')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('equal')
ax.set_aspect('equal')
paths = [path_alt]
# paths, attributes = svg2paths('test.svg')

pts=[]
sample_points_per_seg = 100
for path in paths:
    for seg in path:
        for t in np.linspace(0, 1, sample_points_per_seg):
            p = seg.point(t)
            pts.append(p)
            #ax.plot([x.real for x in pts], [x.imag for x in pts], "-")
            #plt.pause(0.1)

mu = np.mean(pts)
# subtract mean
pts = [x-mu for x in pts]
ax.plot([x.real for x in pts], [x.imag for x in pts], "r-")
plt.pause(1)

print("Calculating coeff...")
N = len(pts)
[freq, coef] = cal_coefficient(N, pts)
print("Done")
n = len(pts)
end_pts_x = []
end_pts_y = []
ax.plot([x.real for x in pts], [x.imag for x in pts], "r-")
ax.axis('equal')
ax.set_aspect('equal')
plt.pause(1)
PLOT = 1
print("# of points:", n)

MaxNpoints = 1000
if n > MaxNpoints:
    indices = np.linspace(0, n-1, MaxNpoints, dtype=np.int32)
    pts = [pts[idx] for idx in indices]

for i, p in enumerate(pts):
    #if i == 0 or i == n-1:
    #    continue
    pos_x = [0]
    pos_y = [0]
    for (f, c) in zip(freq, coef):
        r, phase = cmath.polar(c)
        r /= n
        phase += 2.0 * math.pi * f * i / n
        c = cmath.rect(r, phase)
        x_s = c.real + pos_x[-1]
        x_e = c.imag + pos_y[-1]
        pos_x.append(x_s)
        pos_y.append(x_e)
        if PLOT:
            ax.add_patch(plt.Circle((x_s, x_e), r, alpha=0.3, fill=False))
    end_pts_x.append(pos_x[-1])
    end_pts_y.append(pos_y[-1])
    if PLOT:
        ax.plot([x.real for x in pts], [x.imag for x in pts], "r-")
        ax.plot([p.real], [p.imag], 'r*')
        ax.plot(pos_x, pos_y, '-')
        ax.plot(end_pts_x, end_pts_y, 'y.')
        ax.plot(pos_x[-1], pos_y[-1], 'bo')
        ax.axis('equal')
        ax.set_aspect('equal')
        plt.pause(0.001)
        ax.clear()

plt.plot(end_pts_x, end_pts_y, 'y.')
print("Show")
plt.show()
