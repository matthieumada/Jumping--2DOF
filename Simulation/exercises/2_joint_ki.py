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
Goal: Forward kinematics of a 2-joint robot

(Expected) Result: 10000 (possible) reachable points of its work (i.e., end-effector) space
	joint lengths: L0, L1
	joint angle range: (Minq = np.pi*-0.75, Maxq = np.pi*0.75)

Subtask1: gen_EE(l0, l1, q)
Subtask2: NumSam
Subtask3: sq1
Subtask4: eey
Subtask5: the 2 'for' loops: q[0], q[1], eex[i, j], eey[i, j]
'''

from scipy import *

from matplotlib import *
from pylab import *

import numpy as np
import numpy.linalg as la

#Subtask1
def gen_EE(l0, l1, q):
	EE = np.zeros((2,1))
	
	#EE[0,0] = ?
	#EE[1,0] = ?


	return EE

Minq = np.pi*-0.75
Maxq = np.pi*0.75
#Subtask2
NumSam = 300
sq0 = linspace(Minq,Maxq,NumSam)
#Subtask3
#sq1 = ?
#print size(sq0)
L0 = .31
L1 = .34
q = [np.pi*0.25, np.pi*0.25]

#Work space
eex = np.zeros((NumSam,NumSam))
#Subtask4
#eey = ?

#Subtask5
for i in range(NumSam):
	#q[0] = ?
	for j in range(NumSam):
		#q[1] = ?
		eexy = gen_EE(L0, L1, q)
		#eex[i, j] = ?
		#eey[i, j] = ?

figure(1)
manager = get_current_fig_manager()
manager.window.showMaximized()

title('2-joint-FK', fontsize=60, fontweight='bold')
scatter(eex, eey, marker='o')
xlabel('X', fontsize=55, fontweight='bold')
ylabel('Y', fontsize=55,fontweight='bold', rotation=0)#
tick_params(axis = 'both', which = 'major', labelsize = 55)
legend(loc=2, prop={'size': 55})

#manager = get_current_fig_manager()
#manager.frame.Maximize(True)
savefig('2-joint-fk.png')
show()





