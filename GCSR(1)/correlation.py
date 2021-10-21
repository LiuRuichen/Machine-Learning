import numpy as np
import scipy.io as scio
import seaborn as sns
import matplotlib.pyplot as plt

matFile = r'C:\Users\未央\Desktop\Graph represented-based band selection\五种高光谱数据集\salinas.mat'
mat = scio.loadmat(matFile)

Mat = mat['salinas']

print(Mat.shape)  #145*145*220

[rows,cols,bands] = \
    Mat.shape

A = np.zeros((rows*cols,1))
A = np.array(A)
for i in range(bands):
    band_i = Mat[:,:,i]
    band_i = band_i.reshape(rows*cols, \
                            1, \
                            order='F')
    A = np.hstack((A,band_i))

A = np.delete(A,0,1)

Corr = np.corrcoef(A,rowvar=0)

sns.set()
ax = sns.heatmap(Corr,cmap='rainbow')
plt.show()