import numpy as np
from exercises.integration import euler_explicite 
import roboticstoolbox as rtb 
PI = np.pi

L1 = 0.16 # m
L2 = 0.15 # m
m_leg = 15*10**(-3) # kg
ks = 150.0 # N/m
mq = 2.0 # Ns/m

# Controller parameter 
Kp = 150.0 # N/m stiffnes 
Kd = 0.5 # Ns/m damping 

def forward_kinematics(q):
    q0 = q[0]#-3*PI/4 # fixed 
    q1 = q[1]
    pos = np.zeros((2,1)) #[x,y]
    pos[0] = L1*np.cos(q0) + L2 * np.cos(q0 + q1)
    pos[1] = L1*np.sin(q0) + L2*np.sin(q0 + q1)
    # Jacobian 
    J = np.array([[- L1*np.sin(q0) - L2*np.sin(q0+q1), -L2*np.sin(q0+q1)],
                  [L1*np.cos(q0) + L2*np.cos(q0+q1), L2*np.cos(q0+q1)]])
    l = L2 * np.cos(q0 + q1)
    return  pos, J, l 

def impedance_control(model, d, pos_des, f, torque):
    y_des = 0.3
    # Joint state
    q = d.qpos[[1, 2]]
    print("Joint data=",q, "force=", f)
    dq = d.qvel[[1, 2]]
    
    if f> 10 :
        print("Contact with the ground detected !")
    # Forward kinematics (position of foot)
        pos, J, l = forward_kinematics(q) 

        # virtual mass spring damper system
        y = f/ks + pos[1]
        dpos = J @ dq  # vitesse cartésienne
    # Impedance control law
        F_cmd = Kp*(y_des - y) #+ Kd* dpos 
    # Joint torques using Jacobian transpose
        torque[1] = l * F_cmd /100

        print("Applied torque:", d.ctrl[:])
            
    else:
        print("No contact\t Keep the joint value")
        # print(d.qpos[0], d.qpos[1], d.qpos[2])
        torque[1] = 0.0 # unstable if i keep the toruque from previous step "
    return  torque
    
    



# old postion and jacobian 
    # pos[0] = L1*np.cos(q0) + L2 * np.cos(q0+q1) # x
    # pos[1] = L1*np.sin(q0) + L2*np.sin(q0+q1) # y 

    # J = np.array([[-L1*np.cos(q0) - L2*np.cos(q0+q1), -L2*np.cos(q0+q1)], 
    #               [L1*np.sin(q0) + L2*np.sin(q0+q1), L2*np.sin(q0+q1)]], dtype=np.float64) 
    