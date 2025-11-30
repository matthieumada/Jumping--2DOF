'''
Copyright (C) 2018 Xiaofeng Xiong and Poramate Manoonpong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

'''
Goal: RK4, Eu2, and Odeint solving second order ODE (e.g., a simplified muscle model) 

Application: 'online control' of a simplified muscle model

Subtask1: Runge kutta 4th ODE 'rk4'
Subtask2: Euler ODE 'eu2'
Subtask3: acceleration 'accel'
Subtask4: external force 'f'

'''


from scipy import *

from matplotlib import *
from pylab import *
import numpy as np
from scipy.integrate import odeint

#Subtask1: Runge kutta 4th ODE 'rk4'
def rk4(x, v, dt,f):
    """Returns final (position, velocity) tuple after
    time dt has passed.

    x: initial position (number-like object)
    v: initial velocity (number-like object)
    a: acceleration function a(x,v,dt) (must be callable)
    dt: timestep (number)"""
    x1 = x
    v1 = v
    a1 = accel(x, v, 0, f)

    x2 = x + 0.5*v1*dt
    v2 = v + 0.5*a1*dt
    a2 = accel(x2, v2, dt/2.0,f)

    x3 = x + 0.5*v2*dt
    v3 = v + 0.5*a2*dt
    a3 = accel(x3,v3, dt/2.0,f)

    x4 = x + v3*dt
    v4 = v + a3*dt
    a4 = accel(x4, v4, dt, f)

    xf = x + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
    vf = v + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)

    return xf, vf

def euler(x, v, dt, f):
      a1 = accel(x, v, 0, f)
      # computation
      xf = x + dt*v
      vf = v + dt * a1

      return xf, vf

#Subtask3: acceleration 'accel'
def accel(x, v, dt, f):
    """Determines acceleration from current position,
    velocity, and timestep. This particular acceleration
    function models a spring."""
    stiffness = 2.0
    damping = 2.0
    acceleration = -stiffness*x - damping*v + f
    return acceleration

def dU_dx(U, x):
    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return [U[1], -2*U[1] - 2*U[0] + np.cos(2*x)]

t0 = 0.00
t1= 10.00
tsteps = 100#200
dt = (t1-t0)/tsteps
U0 = [0, 0]
xs = np.linspace(t0, t1, tsteps)#(0, 10, 200)

#rk4
rkpos = zeros(tsteps)
rkid = 0
rkstate = copy(U0)
for t in xs:
	#Subtask4: external force 'f'
	f = cos(2*t)
	rkstate = rk4(rkstate[0], rkstate[1], dt, f)
	rkpos[rkid] = rkstate[0]
	rkid = rkid + 1
#Euler
eupos = zeros(tsteps)
euid = 0
eustate = copy(U0)
for t in xs:
	f = cos(2*t)

	#Subtask2: Euler ODE 'eu2'
	eustate = euler(eustate[0], eustate[1], dt, f)
	eupos[euid] = eustate[0]
	euid = euid + 1
#odeint

Us = odeint(dU_dx, U0, xs)
ys = Us[:,0]

figure(1)
#manager = get_current_fig_manager()
#manager.window.showMaximized()

title('Runge Kutta 4th order', fontsize=60, fontweight='bold')
plot(xs, ys,label = 'ode', linewidth=10, color='b')
plot(xs, rkpos[:],label = 'Rk4', linewidth=10, color='r', linestyle='dashed')
plot(xs, eupos[:],label = 'Eu', linewidth=10, color='g', linestyle=':')
xlabel('Time (s)', fontsize=55, fontweight='bold')
tick_params(axis = 'both', which = 'major', labelsize = 55)
ylabel('Outputs', fontsize=55,fontweight='bold', rotation=0)#  
legend(loc=2, prop={'size': 55})

#manager = get_current_fig_manager()
#manager.frame.Maximize(True)
savefig('rk4.png')
show()





