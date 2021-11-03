import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def data_regularization(path):
    x = []
    y = []
    with open(path,'r') as f:
        data = f.readlines()
        c = 0
        for line in data:
            line = line.strip('\n')
            data[c] = line
            c += 1
            re = line.split(',')
            x.append(re[0])
            y.append(re[-1])
 
    c = 0;
    for x_ in x:
        x_ = float(x_)
        x[c] = x_
        c += 1

    c = 0
    for y_ in y:
        y_ = float(y_)
        y[c] = y_
        c += 1
    
    plt.subplot(1,2,1)
    plt.scatter(x,y,c='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('散点图')
    plt.show()
    
    plt.subplot(1,2,2)
    plt.scatter(x,y,c='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('散点图与拟合曲线')
    plt.show()
    
    data = np.array([x,y])
    data = data.T  
    rows,cols = data.shape  
    bias = np.ones(rows)  
    bias = bias.T   
    data = np.column_stack((bias,data))

    return data
    

def compute_error(X,y,theta):
    sample_size = X.shape[0]
    temp = np.dot(X,theta)-y
    error = 0
    for i in range(0,sample_size):
        error += temp[i]*temp[i]
    error = error / (2*sample_size) 
    return error

def cal_descent(X,y,theta):
    sample_size = X.shape[0]
    diff = np.dot(X,theta)-y
    desc = np.zeros(theta.shape[0])
    desc = desc.T
    
    for j in range(0,theta.shape[0]):
        for i in range(0,sample_size):
            desc[j] += diff[i] * X[i,j]
        desc[j] /= sample_size
    return desc


def update_theta(X,y,theta,alpha):
    desc = cal_descent(X, y, theta)
    for i in range(0,theta.shape[0]):
        theta[i] -= alpha * desc[i]
    
    return theta  


def visualizing(X,y):
    theta_0 = np.linspace(-10, 10, 100)
    theta_1 = np.linspace(-1, 4, 100)
    J_val = np.array([[0 for i in range(theta_1.shape[0])] for j in range(theta_1.shape[0])])
    for i in range(theta_0.shape[0]):
        for j in range(theta_1.shape[0]):
            theta = np.array([theta_0[i],theta_1[j]]).T
            J_val[i,j] = compute_error(X,y,theta)
            
    figure = plt.figure()
    ax = Axes3D(figure)
    X1,Y = np.meshgrid(theta_0,theta_1)
    ax.plot_surface(X1,Y,J_val,rstride=1,cstride=1,cmap='rainbow')
    plt.show()
    
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-1,4)
    levs = np.logspace(0,3,20)
    Contour = plt.contour(X1,Y,J_val,levs,cmap = 'rainbow')
    plt.clabel(Contour,fontsize=10)
    plt.show()
    
    

def main():
    data = data_regularization(r'C:\Users\未央\Desktop\新建文件夹\Linear_Regression.txt')
    X = data[:,0:2]
    y = data[:,2]
    theta = np.zeros(2)
    theta = theta.T
    
    iteration = 1500
    alpha = 0.01
    print(compute_error(X,y,theta))
    
    for i in range(iteration):
        theta = update_theta(X,y,theta,alpha)
        
    print(theta)
      
    X1 = np.linspace(0,25,1000,endpoint = True)
    plt.plot(X1,X1*theta[1]+theta[0],color='black')
    plt.show()
    
    visualizing(X, y)
    
    
if __name__ == '__main__':
    main()