import mujoco
import mujoco.viewer
import numpy as np
import time
import queue
import matplotlib.pyplot as plt
from controller import impedance_control
#from no_mistake import impedance_control 
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
            print("Hip position=", hip_pos)

            adr = m.sensor_adr[sensor_id]       # Id of froce sensor in sensor_data
            dim = m.sensor_dim[sensor_id]       # Dimension of data 
            foot_force = d.sensordata[adr : adr + dim]
            f_norm = np.sqrt(foot_force[0]*foot_force[0] + foot_force[1]*foot_force[1] + foot_force[2]*foot_force[2])
            torque, pos, pos_des, L, Ld = impedance_control(d= d, x_hip=-0.16, z_hip=hip_pos, f_ground=f_norm)
            Force.append(f_norm)
            length.append(L)
            Torque_data.append(torque)
            ground_contact.append(foot_force)
            Joint.append(d.qpos[[1,2]])
            ref_z.append(hip_pos)
            position.append(pos)
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

    plt.subplot(121)
    plt.title("Leg Torque")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.plot(T, np.array(Torque_data)[:,1], label="Leg joinnt")
    plt.legend()
    plt.subplot(122)
    plt.title("Upper leg Torque")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.plot(T, np.array(Torque_data)[:,0], label="Uppee joint")
    plt.legend()
    plt.savefig("./display/Torque")

    plt.figure()
    plt.title("Force measured by sensor over time")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    plt.plot(T, np.array(Force), label="F_ground")
    plt.legend()
    plt.savefig("./display/Force")

    plt.figure()
    plt.title("Joint positions over time")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.plot(T, np.array(Joint)[:,0], label="Upper joint")
    plt.plot(T, np.array(Joint)[:,1], label="Leg joint")
    plt.legend()
    plt.savefig("./display/Joint")

    position = np.array(position)
    length_des = Ld * np.ones(np.shape(position)[0])
    des_x = pos_des[0] * np.ones(np.shape(position)[0]) 
    des_z = pos_des[1] * np.ones(np.shape(position)[0]) 
    ref_z = np.array(ref_z)
    plt.figure()
    plt.subplot(121)
    plt.title("Position on  X axis")
    plt.xlabel("Time [s]")
    plt.ylabel("Axis x [m]")
    plt.plot(T, position[:,0], color='b' ,label="Leg" )
    plt.plot(T, des_x, color='r', label="x goal")
    plt.legend()
    

    plt.subplot(122)
    plt.title("Position on Z axis")
    plt.xlabel("Time [s]")
    plt.ylabel("Axis z [m]")
    plt.plot(T, position[:,1], color='b', label="Leg")
    plt.plot(T, ref_z, color='k', label="hips")
    plt.plot(T, des_z , color='r', label="leg_ref")
    plt.legend()
    plt.savefig("./display/Position")

    plt.figure()
    plt.title("Length in stance phase")
    plt.xlabel("Time [s]")
    plt.ylabel("Length [m]")
    plt.plot(T, length_des, label="L_des")
    plt.plot(T, np.array(length), label="L")
    plt.legend()
    plt.savefig("./display/Length")
    
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
    simulate_free_motion(model_path, sim_time=3.0)
