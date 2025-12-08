import numpy as np
from exercises.integration import euler_explicite 
import roboticstoolbox as rtb 
PI = np.pi

L1 = 0.16 # m
L2 = 0.16 # m
m_leg = 15*10**(-3) # kg
ks = 750000 # N/m
kd = 5000 # Ns/m

# Controller parameter: stiffness and damping
kx = 1.0 * 10 ** 4 # N/m
kz = 1.0 * 10 ** 4 # N/m
K = np.array([kx, kz])
mx = 0 #100 # 2.5 # Ns/m 
mz = 0#100 #2.5 # Ns/m 
M = np.array([mx, mz])

def forward_kinematics(q, dq,  x_hip, z_hip):
    q0 = q[0]#-3*PI/4 # fixed 
    q1 = q[1]

    # take th eposiiton of the hip from the model
    pos_foot = np.zeros((2,1)) #[x,z]
    delta_x = L1*np.sin(q0) + L2*np.sin(q0 + q1)
    delta_z = L1*np.cos(q0) + L2*np.cos(q0 + q1)
    pos_foot[0] = x_hip + delta_x #x
    pos_foot[1] = z_hip + delta_z #z 

    # Jacobian 
    J = np.array([[L1*np.cos(q0) + L2*np.cos(q0+q1), L2*np.cos(q0+q1)],
                  [-L1*np.sin(q0) - L2*np.sin(q0+q1),  -L2*np.sin(q0+q1)]])
    
    # lenght of the vritual model 
    L = np.sqrt( (delta_x)**2 + (delta_z)**2 )
    L_point = - (2*L1*L2/L)*np.sin(q1) * dq[1]
    return  pos_foot, J, L, L_point, delta_x, delta_z

def impedance_control(d, x_hip, z_hip, f_ground):
    Ld = 0.25 # desired leg length with all leg have a angle of pi/4 
    pos_des = np.array([ x_hip, z_hip])

    # Joint state
    q = d.qpos[[1, 2]]
    dq = d.qvel[[1, 2]]
    # Forward kinematics (position of foot)
    pos_foot, J, L, L_point, delta_x, delta_z = forward_kinematics(q, dq, x_hip, z_hip) 

    if f_ground > 50:
        print("Ground reaction force due to contact =", f_ground,"--> Stance phase")
        # virtual force control 
        F_leg = - ks*(L - Ld ) - kd*L_point
        F = F_leg * np.array([ delta_x/L, delta_z/L])  # convert to cartesian force
        torque =  J.T @ F # joint torques
        print("Torque in stance leg =", torque)
        return torque, pos_foot
    else:
        print("No contact with the ground --> Swing phase")
        # PD position control 
        F_swing = - K @ (pos_foot - pos_des) -  M @ J[0, :]
        torque = J.T @ F_swing
        print("Torque in swing leg =", torque, "position_real=", pos_foot, "position_des=", pos_des)
        return torque, pos_foot
    
    