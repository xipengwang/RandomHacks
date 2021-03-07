import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

assert(len(sys.argv) > 1)
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(221)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])


print(img.shape)
(rows, cols) = img.shape

def local_maximum_(r, c):
    def in_image_range_(r, c):
        return r >= 0 and r < rows and c >=0 and c < cols

    candidates = [(r-1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1), (r, c+1), (r+1, c-1), (r+1, c), (r+1, c+1)]
    for e in candidates:
        if in_image_range_(e[0], e[1]) and magnitude_spectrum[r, c] < magnitude_spectrum[e[0], e[1]]:
            return False
    return True

crow, ccol = int(rows/2) , int(cols/2)

# for r in range(0, rows):
#     for c in range(0, cols):
#         print(r, c)
#         if local_maximum_(r, c):
#             fshift[r, c] = 0
#             magnitude_spectrum[r, c] = 0

K1 = 20
K2 = 80

import math
t = np.linspace(0, 2*math.pi, 1000)

for e in t:
    for k in range(K1,K2+1):
        r = crow+int(k * math.cos(e))
        c = ccol+int(k * math.sin(e))
        fshift[r, c] = 0
        magnitude_spectrum[r, c] = 0


plt.subplot(223)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum after filtering')
plt.xticks([]), plt.yticks([])

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.subplot(224)
plt.imshow(img_back, cmap = 'gray')
plt.title('Image after filtering')
plt.xticks([]), plt.yticks([])

plt.show()
