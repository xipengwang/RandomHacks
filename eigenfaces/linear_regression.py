import jax.numpy as jnp
from jax import grad, random
from matplotlib import pyplot as plt
import numpy as np


X = [(1,1), (1,2), (2,1), (2,2)]
Y = [(-1,-1), (-1,-2), (-2,-1), (-2,-2)]


inputs = jnp.array([[X[0][0], X[0][1]], [X[1][0], X[1][1]], [X[2][0], X[2][1]], [X[3][0], X[3][1]],
                    [Y[0][0], Y[0][1]], [Y[1][0], Y[1][1]], [Y[2][0], Y[2][1]], [Y[3][0], Y[3][1]]
]).T


targets = jnp.array([True, True, True, True, False, False, False, False])


def predict(W, b, inputs):
    outputs = jnp.dot(W, inputs) + b
    return outputs

def loss_fun(W, b, inputs, targets):
  preds = predict(W, b, inputs)
  loss = 0
  for i, t in enumerate(targets):
      if t == True:
          loss += jnp.max(jnp.array([preds[1][i] - preds[0][i] + 1, 0]))
      else:
          loss += jnp.max(jnp.array([preds[0][i] - preds[1][i] + 1, 0]))
  t = 0
  return loss / targets.shape[0] + t * jnp.linalg.norm(W)

key = random.PRNGKey(1024)
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (2,2))
BIAS=False
if BIAS:
    b = random.normal(b_key, (2,1))
else:
    b = jnp.zeros((2,1))
lr = 0.1
loss_prev = -1
PLOT=True
for i in range(1000):
    print(f"iter: {i}.................")
    grad_func = grad(loss_fun, (0, 1))
    W_grad, b_grad = grad_func(W, b, inputs, targets)
    loss = loss_fun(W, b, inputs, targets)
    preds = predict(W, b, inputs)
    classification = preds[0, :] > preds[1, :]
    print(f"      loss: {loss}")
    print(f"prediction: {classification}")
    print(f"   targets: {targets}")

    # plot W[0][0]*x + W[0][1]*y + b[0] = 0
    plt.plot([-3, 3], [(-b[0]+3*W[0][0])/W[0][1], (-b[0]-3*W[0][0])/W[0][1]], 'r-')
    # plot W[1][0]*x + W[1][1]*y + b[1] = 0
    plt.plot([-3, 3], [(-b[1]+3*W[1][0])/W[1][1], (-b[1]-3*W[1][0])/W[1][1]], 'g-')
    plt.plot([e[0] for e in X], [e[1] for e in X], 'ro')
    plt.plot([e[0] for e in Y], [e[1] for e in Y], 'g*')
    if PLOT:
        for x in jnp.linspace(-2, 2, 10):
            for y in jnp.linspace(-2, 2, 10):
                preds = predict(W, b, jnp.array([[x],[y]]))
                c = preds[0, 0] > preds[1, 0]
                if c==True:
                    plt.plot([x], [y], 'r.')
                else:
                    plt.plot([x], [y], 'g.')
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.pause(0.001)
    W -= lr * W_grad
    if BIAS:
        b -= lr * b_grad
    if jnp.abs(loss - loss_prev) < 1E-5:
        break
    loss_prev = loss
    plt.clf()

plt.show()
