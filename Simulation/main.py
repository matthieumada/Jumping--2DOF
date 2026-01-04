import mujoco
import mujoco.viewer
import numpy as np
import time
import queue
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

from controller import impedance_control
from controller_lqr import impedance_lqr

PI = np.pi

# --- Main function ---
def simulate_free_motion(model_path, sim_time):

    # data
    Joint = []
    Torque_data= []
    Force = []
    ground_contact = []
    position = []
    ref_z = [] 
    length = []
    goal = []
    # load model 
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    # Initial paramter for only one control 
    d.qpos[:] = np.array([0,-PI/4, PI/2]) # position:[hip(m) , upper(rd), leg(rd)]
    d.qvel[:] = np.zeros(m.nv)  # speed 

    # update phyics computation (kinematics, dynamics, sensors, contacts without move th eismulation 
    mujoco.mj_forward(m, d) 

    # Collect data
    timestep = m.opt.timestep
    steps = int(sim_time / timestep)
    T = np.linspace(0, sim_time, steps)
    print("Model=", model_path)
    print("Timestep =",timestep, "steps =",steps)
    print("Sensor list :", [m.sensor(i).name for i in range(m.nsensor)])
    #Initialization viewer
    torque = np.zeros((2) ) # two joints
    with mujoco.viewer.launch_passive(model=m, data=d) as viewer:
        sim_start = time.time()
        sensor_id = m.sensor('socle_contact').id
        hip_sensor_id = m.sensor('hip_position').adr

        for i in range(steps):
            # forces data 
            hip_pos =float(d.sensordata[hip_sensor_id])
            #print("Hip position=", hip_pos)

            adr = m.sensor_adr[sensor_id]       # Id of froce sensor in sensor_data
            dim = m.sensor_dim[sensor_id]       # Dimension of data 
            foot_force = d.sensordata[adr : adr + dim]
            f_norm = np.sqrt(foot_force[0]*foot_force[0] + foot_force[1]*foot_force[1] + foot_force[2]*foot_force[2])
            #torque, pos, pos_des, L, Ld = impedance_control(d= d, x_hip=-0.16, z_hip=hip_pos, f_ground=f_norm)
            torque, pos, pos_des, L, Ld = impedance_lqr(m=m, d= d, x_hip=-0.16, z_hip=hip_pos, f_ground=f_norm)
            Force.append(f_norm)
            length.append(L)
            Torque_data.append(torque)
            ground_contact.append(foot_force)
            Joint.append(d.qpos[[1,2]])
            ref_z.append(hip_pos)
            position.append(pos)
            goal.append(pos_des)
            # Move to a new state of the simulation 
            d.ctrl = torque
            mujoco.mj_step(m, d) 

            # synchronization viewer
            viewer.sync()

            # If viewer closed -> get out
            if not viewer.is_running():
                break

            # Option : real time system
            time.sleep(timestep)
        print("Simulation finished. Viewer stays open. Close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
    # Mujuco changes the style of plt so add it after the end of simulation 
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    ax1.set_title("Leg Torque", fontsize=16)
    ax1.set_xlabel("Time [s]", fontsize=14)
    ax1.set_ylabel("Torque [Nm]", fontsize=14)
    ax1.plot(T, np.array(Torque_data)[:,1], label="Leg joint", linewidth=2)
    ax1.legend(loc='best', fontsize=12)

    ax2.set_title("Upper leg Torque", fontsize=16)
    ax2.set_xlabel("Time [s]", fontsize=14)
    ax2.set_ylabel("Torque [Nm]", fontsize=14)
    ax2.plot(T, np.array(Torque_data)[:,0], label="Upper joint", linewidth=2)
    ax2.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig("./display/Torque_LQR")

    plt.figure(figsize=(10, 6))
    plt.title("Force measured by sensor over time", fontsize=16)
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("F [N]", fontsize=14)
    plt.plot(T, np.array(Force), label="F_ground", linewidth=2)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig("./display/Force_LQR")
    
    plt.figure(figsize=(10, 6))
    plt.title("Joint positions over time", fontsize=16)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Angle [rad]", fontsize=12)
    plt.plot(T, np.array(Joint)[:,0], label="Upper joint", linewidth=2)
    plt.plot(T, np.array(Joint)[:,1], label="Leg joint", linewidth=2)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig("./display/Joint_LQR")

    position = np.array(position)
    goal = np.array(goal)
    length_des = Ld * np.ones(np.shape(position)[0])
    des_x = pos_des[0] * np.ones(np.shape(position)[0]) 
    des_z = pos_des[1] * np.ones(np.shape(position)[0]) 
    ref_z = np.array(ref_z) # hip position 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    ax1.set_title("Position on X axis", fontsize=16)
    ax1.set_xlabel("Time [s]", fontsize=14)
    ax1.set_ylabel("Axis X [m]", fontsize=14)
    ax1.plot(T, position[:,0], color='b' ,label="Leg" , linewidth=2)
    ax1.plot(T, goal[:,0], color='r' ,label="x_goal" , linewidth=2)
    ax1.legend(loc='best', fontsize=12)

    ax2.set_title("Position Z axis", fontsize=16)
    ax2.set_xlabel("Time [s]", fontsize=14)
    ax2.set_ylabel("Axis Z [m]", fontsize=14)
    ax2.plot(T, position[:,1], color='b', label="Leg", linewidth=2)
    ax2.plot(T, ref_z, color='k', label="hips", linewidth=2)
    ax2.plot(T, goal[:,1] , color='r', label="leg_ref", linewidth=2)
    ax2.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig("./display/Position_LQR")

    plt.figure(figsize=(10, 6))
    plt.title("Length in stance phase", fontsize=18)
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Length [m]", fontsize=14)
    plt.plot(T, length_des, linewidth=2, label="L_des")
    plt.plot(T, np.array(length), linewidth=2, label="L")
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig("./display/Length_LQR")

    print("Model=", model_path)
    print("Timestep =",timestep, "steps =",steps)
    print("Sensor list :", [m.sensor(i).name for i in range(m.nsensor)])
    print("Actuators:", m.nu)
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print("id=",i,"name_actuator=",name)
    print("force without contact", Force[-1])
    print("x desired=", des_x[-1], "real_x=",position[:,1][-1], "difference=", des_x[-1] - position[:,1][-1])
    return 

# --- Main launcher  ---
if __name__ == "__main__":
    model_path = "ref_stand.xml"  # XML file
    simulate_free_motion(model_path, sim_time=2.5)
