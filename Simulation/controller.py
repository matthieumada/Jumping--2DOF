import numpy as np
from exercises.integration import euler_explicite 
import roboticstoolbox as rtb 
PI = np.pi

L1 = 0.16 # m
L2 = 0.16 # m
m_leg = 15*10**(-3) # kg

# Virtual model parameter 
ks = 7500 # N/m
kd = 0.5 # Ns/m

#Controller parameter: stiffness and damping PD swing leg 
kx = 1.75 * 10 **(3)# N/m
kz = 5.0 * 10 **(2) # N/m
mx = 1.0 * 10 **(1) # Ns/m 
mz = 1.0 * 10 **(1) # Ns/m 

# Controller parameter : stiffness swing P leg
# kx = 1.75 * 10 **(3)# N/m
# kz = 5.0 * 10 **(2) # N/m
# mx = 0  # Ns/m 
# mz = 0 # Ns/m 

def forward_kinematics(q, dq,  x_hip, z_hip):
    q0 = float(q[0])
    q1 = float(q[1])

    delta_x = L1*np.sin(q0) + L2*np.sin(q0 + q1)
    delta_z = L1*np.cos(q0) + L2*np.cos(q0 + q1)

    # take the positon of the hip from the model
    pos_foot = np.zeros((2,1)) # [x,z]
    pos_foot[0] = x_hip - delta_x #x
    pos_foot[1] = z_hip - delta_z #z 

    # Jacobian 
    J = np.array([[-L1*np.cos(q0) - L2*np.cos(q0+q1), -L2*np.cos(q0+q1)],
                  [ L1*np.sin(q0)  + L2*np.sin(q0+q1),  L2*np.sin(q0+q1)]])
    
    # lenght of the vritual model 
    L = np.sqrt( (delta_x)**2 + (delta_z)**2 )
    L_point = - (2*L1*L2/L)*np.sin(q1) * dq[1]
    return  pos_foot, J, L, L_point, delta_x, delta_z

def impedance_control(d, x_hip, z_hip, f_ground):
    Ld = 0.2 # desired leg length with all leg have a angle of pi/4 
    pos_des = np.array([ x_hip, z_hip - 0.2])

    # Joint state
    q = d.qpos[[1, 2]]
    dq = d.qvel[[1, 2]]
    # Forward kinematics (position of foot)
    pos_foot, J, L, L_point, delta_x, delta_z = forward_kinematics(q, dq, x_hip, z_hip) 
    if f_ground > 23.15:
        print("Ground reaction force due to contact =", f_ground,"--> Stance phase")
        # virtual force control 
        print("L=",L)
        F_leg = - ks*(L - Ld ) - kd*L_point
        F = F_leg * np.array([ delta_x/L, delta_z/L])  # convert to cartesian force
        torque =  J.T @ F # joint torques
        print("Torque in stance leg =", torque)
    else:
        #print("No contact with the ground --> Swing phase")
        # PD position control 
        F_swing = np.array((2,1))

        # computation by hand
        F_swing[0] = - kx*(pos_foot[0] - pos_des[0]) - mx * (J[0,0] *dq[0] - J[0,1] *dq[1])
        F_swing[1] = - kz*(pos_foot[1] - pos_des[1]) - mz * (J[1,0] *dq[0] - J[1,1] *dq[1])

        torque = J.T @ F_swing
        #print("Torque in swing leg =", torque, "position_real=", pos_foot, "position_des=", pos_des)
    return torque, pos_foot, pos_des, L, Ld
    
    