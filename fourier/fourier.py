from svgpathtools import svg2paths, parse_path
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math


def cal_coefficient(N, pts):
    # N >= 1:
    freq = [0]
    freq.extend(list(range(-N, 0, 1)))
    freq.extend(list(range(1, N+1, 1)))
    size = len(pts)
    (time, delta_t) = np.linspace(0, 1, size, retstep=True)
    coef = []
    for f in freq:
        c = complex(0, 0)
        for p, t in zip(pts, time):
            r, phase = cmath.polar(p)
            r *= delta_t
            phase -= f * 2 * math.pi * t
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

N = 100
pts=[]
for path in paths:
    for seg in path:
        for t in np.linspace(0, 1, 100 if N < 100 else N):
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
[freq, coef] = cal_coefficient(N, pts)
print("Done")
size = len(pts)
(time, delta_t) = np.linspace(0, 1, size, retstep=True)
end_pts_x = []
end_pts_y = []
ax.plot([x.real for x in pts], [x.imag for x in pts], "r-")
ax.axis('equal')
ax.set_aspect('equal')
plt.pause(1)
PLOT = 1
print("# of points:", size)
prev_t = -1
for i, (p, t) in enumerate(zip(pts, time)):
    if i == 0 or i == size-1:
        prev_t = t
        continue
    if (t - prev_t) < 0.005:
        continue;
    prev_t = t
    pos_x = [0]
    pos_y = [0]
    for (f, c) in zip(freq, coef):
        r, phase = cmath.polar(c)
        phase += f * 2 * math.pi * t
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
