import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

assert(len(sys.argv) > 2)
img1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)

assert(img1.shape == img2.shape)
print(img1.shape, img2.shape)

f = np.vstack((img1.reshape(-1), img2.reshape(-1)))
mean_f = np.mean(f, axis=0)
norm_f = f - mean_f

[U, S, Vh] = np.linalg.svd(norm_f.T, full_matrices=False)
e_img1 = U[:, 0].reshape(img1.shape)
e_img2 = U[:, 1].reshape(img2.shape)
train_weights = np.diag(S).dot(Vh)
#print(f"Train weights: {train_weights}")

test_weight = U.T.dot(test_img.reshape(-1) - mean_f)
#print(f"Test weights: {test_weight}")

dist = train_weights - test_weight.reshape(2,1)
print(dist)
score = np.linalg.norm(dist, axis=0)
print(f"Score: {score}")



plt.subplot(321)
plt.imshow(img1, cmap = 'gray')
plt.title('training face1')
plt.xticks([]), plt.yticks([])

plt.subplot(322)
plt.imshow(img2, cmap = 'gray')
plt.title('training face2')
plt.xticks([]), plt.yticks([])


plt.subplot(323)
plt.imshow(e_img1, cmap = 'gray')
plt.title('eigen face1')
plt.xticks([]), plt.yticks([])

plt.subplot(324)
plt.imshow(e_img2, cmap = 'gray')
plt.title('eigen face2')
plt.xticks([]), plt.yticks([])

plt.subplot(325)
plt.imshow(test_img, cmap = 'gray')
plt.title('Test image')
plt.xticks([]), plt.yticks([])


plt.subplot(326)
if score[0] < score[1]:
    plt.imshow(img1, cmap = 'gray')
    print(f"\nSame class as {sys.argv[1]}\n")
else:
    plt.imshow(img2, cmap = 'gray')
    print(f"\nSame class as {sys.argv[2]}\n")


plt.title('Classification result')
plt.xticks([]), plt.yticks([])



plt.show()
