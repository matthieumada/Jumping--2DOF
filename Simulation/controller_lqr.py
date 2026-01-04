import numpy as np
import mujoco
import scipy.linalg
PI = np.pi

L1 = 0.16 # m
L2 = 0.16 # m
m_leg = 15*10**(-3) # kg

# Virtual model parameter 
ks = 7500 # N/m
kd = 0.5 # Ns/m


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
    
    # lenght of the virtual model 
    L = np.sqrt( (delta_x)**2 + (delta_z)**2 )
    L_point = - (2*L1*L2/L)*np.sin(q1) * dq[1]
    return  pos_foot, J, L, L_point, delta_x, delta_z

# Compute of K to avoid to do it at each call
# Initalize a None
CACHED_K = None

def compute_lqr_gain(M_inv):
    # State-space matrices
    A = np.zeros((4, 4))
    A[0, 2] = 1; A[1, 3] = 1
    
    B = np.zeros((4, 2))
    B[2:, :] = M_inv
    
    Q = np.diag([500, 500, 10, 10]) 
    R = np.diag([10, 10])
    
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ P)
    return K

def lqr_control(m, d, q_des, dq_des):
    global CACHED_K
    
    idx = [1, 2] 
    
    # Mass matrix
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    
    # Extract hip and knee
    M_joint = M[np.ix_(idx, idx)] # Manière propre numpy d'extraire [1,2]x[1,2]
    M_inv = np.linalg.inv(M_joint)
    
    # K computed only if K is None
    # This avoids recomputing K at each call
    if CACHED_K is None:
        CACHED_K = compute_lqr_gain(M_inv)

    # State error
    q_curr = d.qpos[idx]
    dq_curr = d.qvel[idx]
    
    x_err = np.concatenate((q_curr - q_des, dq_curr - dq_des))
    
    # 5. Command u = -Kx
    tau_lqr = -CACHED_K @ x_err 
    
    # Coriolis effect and gravity compensation
    tau_gravity = d.qfrc_bias[idx]
    
    return tau_lqr + tau_gravity

def impedance_lqr(m, d, x_hip, z_hip, f_ground):
    Ld = 0.2 # desired leg length with all leg have a angle of pi/4 
    pos_des = np.array([ x_hip, z_hip - 0.2])

    # Joint state
    q = d.qpos[[1, 2]]
    dq = d.qvel[[1, 2]]
    
    pos_foot, J, L, L_point, delta_x, delta_z = forward_kinematics(q, dq, x_hip, z_hip) 
    
    if f_ground > 23.5: 
        print("Ground reaction force due to contact =", f_ground,"--> Stance phase")
        # virtual force control 
        print("L=",L)
        F_leg = - ks*(L - Ld ) - kd*L_point
        F = F_leg * np.array([ delta_x/L, delta_z/L])  # convert to cartesian force
        torque =  J.T @ F # joint torques
        print("Torque in stance leg =", torque)
        
    else:
        # Swing phase
        
        # Shape target in the air
        q_target = np.array([-PI/4, PI/2]) 
        dq_target = np.zeros(2)
        
        torque_lqr = lqr_control(m, d, q_des=q_target, dq_des=dq_target)
        
        # torque applied
        torque = torque_lqr

    return torque, pos_foot, pos_des, L, Ld