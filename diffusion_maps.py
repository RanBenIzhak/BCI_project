
import numpy as np
from numpy import linalg as LA
import os, math

newDim = 64


def normalize(arr):
    arr = arr.astype('float32')
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def weightedAverage(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def diffusionMapping(data, k, t, **kwargs):
    try:
        kwargs['dim'] or kwargs['delta']
    except KeyError:
        raise KeyError('specify either dim or delta as keyword argument!')

    dataList = []  # create list whose indices will serve as references for the vectors from now on
    for x in data:
        dataList.append(x)
    X = range(len(dataList))

    # construct Markov matrix
    v = []
    for x in X:  # for each vector in data
        vx = 0
        for y in X:  # compare with every other vectors in data
            _x = np.array(dataList[x])
            _y = np.array(dataList[y])
            vx += k(_x, _y)
        v.append(math.sqrt(vx))    # v_i is sum_over_j(sqrt( k(x_i, y_j) ) )

    p = []
    for x in X:
        p.append([])
        for y in X:
            _x = np.array(dataList[x])
            _y = np.array(dataList[y])
            p[x].append(k(_x, _y) / (v[x] * v[y]))

    # compute eigenvectors of (a_ij)
    phi = []
    eigval, eigvec = LA.eigh(np.array(p))
    for i in range(len(eigvec)):
        phi.append(eigvec[:, i])
    # reverse order
    eigval[:] = eigval[::-1]
    phi[:] = phi[::-1]

    # compute dimension
    # (for better performance you may want to combine this with an iterative way of computing eigenvalues/vectors)
    if kwargs['dim']:
        embeddim = kwargs['dim']
    elif kwargs['delta']:
        i = 1
        while eigval[i] ** t > kwargs['delta'] * eigval[1] ** t:
            i += 1
        embeddim = i

    # compute embedding coordinates
    Psi = []
    for x in X:
        Psi.append([])
        for j in range(embeddim):
            i = j + 1  # ignore the first eigenvector/value as this is only constant
            Psi[x].append((eigval[i] ** t) * phi[i][x] / v[x])
    return (Psi, dataList)



# showImages = False
#
# coordinates, dataList = diffusionMapping(data, lambda x, y: math.exp(-LA.norm(x - y) / 1024), 1, dim=2)
# a = np.asarray(coordinates)
# x = a[:, 0]
# y = a[:, 1]
# fig, ax = plt.subplots()
# j = 0
# if showImages:
#     squareLength = math.sqrt(len(dataList[0]))
#     square = (squareLength, squareLength)
#     for xpt, ypt in zip(x, y):
#         img = np.array(dataList[j]).reshape(square)[::2, ::2]
#         ab = AnnotationBbox(OffsetImage(img, cmap=cm.Greys_r), [xpt, ypt],
#                             xybox=(65., 0),
#                             xycoords='data',
#                             boxcoords="offset points",
#                             frameon=False,
#                             arrowprops=dict(arrowstyle="->"))
#         ax.add_artist(ab)
#         j = j + 1
# else:
#     labels = ['image {0}'.format(i + 1) for i in range(len(x))]
#     for label, xpt, ypt in zip(labels, x, y):
#         plt.annotate(
#             label,
#             xy=(xpt, ypt), xytext=(-20, 20),
#             textcoords='offset points', ha='right', va='bottom',
#             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
#             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
# ax.plot(x, y, 'ro')
# plt.show()