# file for integration design 
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt 

def func(x):
    #return x**(4) + x**2 - 8*x + 1 
    return np.sin(x)
 
def Primi(x):
    #return (1/5)*x**(5) + (1/3)*x**(3) - 4*x**(2) + x 
    return -np.cos(x)

def euler_explicite(ddx_pred, x_pred, dt):
    x = x_pred + ddx_pred * dt 
    return x


def Verlet(ddx, ddx_pred, x_pred, dt):
    return x_pred + 0.5 * ( ddx+ddx_pred)*dt 


def main():
     Euler = [Primi(-4)]
     verlet = [Primi(-4)]
     adam = [Primi(-4)]
     T = np.linspace(-4,4,200)
     dt = T[1]-T[0]
     x = [func(i) for i in T]
     primi = [Primi(i) for i in T]
     for j in range(1,len(T)):
         Euler.append(euler_explicite(x[j-1], Euler[j-1],dt))
         verlet.append(Verlet( x[j], x[j-1], Euler[j-1], dt))
         
     plt.figure()
     plt.title("Integration comparison of speed ")
     plt.xlabel(" Time [s]")
     plt.ylabel("Value ")
     plt.plot(T,primi, label="Primitive")
     plt.plot(T,Euler, label="Euler")
     plt.plot(T,verlet, label="Verlet")
     plt.grid("True")
     plt.legend()

     plt.show()


if __name__ == "__main__":
    main()
