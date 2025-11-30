import mujoco
import mujoco.viewer
import numpy as np
import time
import queue
import matplotlib.pyplot as plt
from controller import impedance_control
PI = np.pi

# --- Main function ---
def simulate_free_motion(model_path, sim_time):
    # load model 
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    # goal 
    pos_des = np.array([-0.05, 0.7])
    
    # Initial paramter for only one control 
    d.qpos[:] = np.array([-0.3,-PI/4, PI/4]) # position:[hip (m) , upper(rd), leg(rd)]
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
            #print(sensor_id) 
            hip_pos = d.sensordata[hip_sensor_id]
            adr = m.sensor_adr[sensor_id]       # Id of froce sensor in sensor_data
            dim = m.sensor_dim[sensor_id]       # Dimension of data 
            foot_force = d.sensordata[adr : adr + dim]
            f_norm = np.sqrt(foot_force[0]*foot_force[0] + foot_force[1]*foot_force[1] + foot_force[2]*foot_force[2])
            torque = impedance_control(model=m, d=d, pos_des=pos_des, f=f_norm, torque=torque)
            #print("Force sol sur pied (N) :",foot_force)
            #print("Norm force", f_norm 
            
            # Free system
            #d.ctrl[:] = 0.0
            # controller
            #pos = impedance_control(model=m, data=d, pos_des=pos_des, f=f_norm)
            # pos_computed.append(pos)
            # pos_desired.append(pos_des[1])
            # pos_measured.append(hip_pos)
            d.ctrl[:] = torque 
            # Move to a new state of the simulation 
            mujoco.mj_step(m, d) 

            # synchronization viewer
            viewer.sync()

            # If viewer closed -> get out
            if not viewer.is_running():
                break

            # Option : real time system
            time.sleep(timestep)
    print(m.actuator(0).name)
    print(m.actuator(1).name)


    return 


# --- Main launcher  ---
if __name__ == "__main__":
    model_path = "stand_complete.xml"  # XML file
    simulate_free_motion(model_path, sim_time=3.0)
