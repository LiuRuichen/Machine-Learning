import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
z = np.array([[0 for i in range(100)]for j in range(100)])
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        z[i,j] = x[i]*x[i] + y[j]*y[j]
        
figure = plt.figure()
ax = Axes3D(figure)
X,Y = np.meshgrid(x,y)
ax.plot_surface(X,Y,z,rstride=1,cstride=1,cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

f1= plt.figure()
plt.xlim(-10,10)
plt.ylim(-10,10)
levs = np.logspace(0,3,30)
Contour = plt.contour(X,Y,z)
plt.clabel(Contour,fontsize=10)

iteration = 1500
x_ini = -7.86
y_ini = 8.63
alpha = 0.001
#plt.scatter(x_ini,y_ini,marker='x',c='red',linewidths=3)
x = x_ini
y = y_ini

xx = []
yy = []
xx.append(x_ini)
yy.append(y_ini)
for i in range(iteration):
    x = x - alpha * 2 * x
    y = y - alpha * 2 * y
    if i%100 == 0:
        xx.append(x)
        yy.append(y)

xx = np.array(xx)
yy = np.array(yy)

def update_points(num):
    point_ani.set_data(xx[num], yy[num])
    text_pt.set_position((xx[num], yy[num]))
    text_pt.set_text("x=%.3f, y=%.3f"%(xx[num], yy[num]))
    return point_ani,text_pt
 


point_ani, = plt.plot(xx[0], yy[0], "r+")


text_pt = plt.text(0,0,'',fontsize=10)
# 开始制作动画
ani = animation.FuncAnimation(f1, update_points, np.arange(xx.shape[0]), interval=300, blit=True)

plt.xlabel('x')
plt.ylabel('y')
plt.title('contour and the tendency')
ani.save(r'./test.gif', writer='pillow', fps=5)

plt.show()


