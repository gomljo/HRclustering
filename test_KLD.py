from HR_TREE.HRHC import *
from HR_TREE.make_data import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr


# index, X, y = generate(type_='blobs', n_cluster=2, n_features=2, n_samples=500, n_clusters_per_class=1)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# # print(X)
# # print(X.shape)
# # plot_data(X, y)
# cl_model = HRHC(X=X, index=index, tau=0.04)
# cl_model.fit_predict(min_samples=3)

def KLD(dist1, dist2, dx):
    return np.sum(np.where(dist1 != 0, dist1 * np.log(dist1 / dist2) * dx, 0))


def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))


def test_gauss_pdf():
    x = np.array([[0],[0]])
    mu  = np.array([[0],[0]])
    cov = np.eye(2)

    print(pdf_multivariate_gauss(x, mu, cov))

    # prints 0.15915494309189535
mu1 = 0.21899386
var1 = 0.00115031
# var1 = 1

mu2 = 0.15524015
var2 = 0.00044619
# var2 = 1

mu3 = 0.18517339
var3 = 0.00106302

mu4 = 0.23523881
var4 = 0.00108384

min12 = 0.1178778
max12 = 0.2728686

min13 = 0.1178778
max13 = 0.2728686

min14 = 0.1178778
max14 = 0.29828935

x_range = np.linspace(min14, max14, 1000)

y1 = (1/np.sqrt(2*np.pi*var1)) * np.exp(-1 * ((x_range - mu1)**2 / (2*var1)))
y2 = (1/np.sqrt(2*np.pi*var2)) * np.exp(-1 * ((x_range - mu2)**2 / (2*var2)))
y3 = (1/np.sqrt(2*np.pi*var3)) * np.exp(-1 * ((x_range - mu3)**2 / (2*var3)))
y4 = (1/np.sqrt(2*np.pi*var4)) * np.exp(-1 * ((x_range - mu4)**2 / (2*var4)))
plt.plot(x_range, y1, c='red', label='hr1')
plt.plot(x_range, y2, c='blue', label='hr2')
plt.plot(x_range, y3, c='green', label='hr3')
plt.plot(x_range, y4, c='orange', label='hr4')
plt.legend()
plt.show()

# print(np.sum(y1)*((max14-min14)/1000))
# # print(y1)
# print(np.sum(y2)*((max14-min14)/1000))
# # print(y3)
# print(np.sum(y3)*((max14-min14)/1000))
kld1 = KLD(y1, y2, ((max12-min12)/1000))
# kld2 = KLD(y1, y2, ((max12-min12)/1000))
kld3 = KLD(y1, y3, ((max12-min12)/1000))
kld4 = KLD(y1, y4,((max14-min14)/1000))

print(kld1)
print(np.exp(-kld1))

print(kld3)
print(np.exp(-kld3))

print(kld4)
print(np.exp(-kld4))

print('KL-divergence(box_1 || box_2): %.3f ' % sum(rel_entr(y1,y4)))

