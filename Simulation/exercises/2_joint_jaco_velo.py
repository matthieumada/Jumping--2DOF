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
Goal: Jacobian matrix of a 2-joint robot

(Expected) Result: calculating the velocities of its end-effector given by the joint angles (i.e., q) and velocities (i.e., dq)

Subtask1: gen_jacEE(l0, l1, q)
Subtask2: eev
'''

from scipy import *

from matplotlib import *
from pylab import *

import numpy as np
import numpy.linalg as la

#Subtask1:
def gen_jacEE(l0, l1, q):
	JEE = np.zeros((6,2))
	# Position part of jacobian 
	JEE[0,0] = -L0 * np.sin(q[0]) - L1 * np.sin(q[0] + q[1])
	JEE[0,1] = -L1 * np.sin(q[0] + q[1])

	JEE[1,0] = L0 * np.cos(q[0]) + L1*np.cos(q[0] + q[1])
	JEE[1,1] = L1*np.cos(q[0] + q[1])

	# rotation part of jacobian 
	JEE[5,0] = 1
	JEE[5,1] = 1

	return JEE


#Jacobian - dq -> dx
L0 = 1.0
L1 = 1.0

# joint angular position
q = [np.pi/4.0, np.pi*3.0/8.0]

# derivate of joints
dq = [np.pi/10.0, np.pi/10.0]

dq = np.asmatrix(dq)
dq = dq.T
#print dq.shape

Jee = np.asmatrix(gen_jacEE(L0, L1, q))

#Subtask2:
eev = Jee *  dq

# print 'Its linear and angular velocities are', eev #python 2.x
print ('Its linear and angular velocities are', eev)#python 3.x






