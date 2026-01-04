import mujoco
import numpy as np
import matplotlib.pyplot as plt
import copy

# Importation de tes modules de contrôle
# Assure-toi que controller.py et controller_lqr.py sont dans le même dossier
import controller      
import controller_lqr  

plt.style.use('seaborn-v0_8-whitegrid')
PI = np.pi

def run_simulation(model_path, control_func, label, sim_time=2.5):
    """
    Exécute une simulation complète avec une fonction de contrôle donnée.
    """
    print(f"--- Running Simulation for: {label} ---")
    
    # Chargement du modèle
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    # Conditions initiales
    d.qpos[:] = np.array([0, -PI/4, PI/2]) 
    d.qvel[:] = np.zeros(m.nv)
    mujoco.mj_forward(m, d)
    
    steps = int(sim_time / m.opt.timestep)
    # Initialisation du dictionnaire de données
    data = {
        "time": np.linspace(0, sim_time, steps),
        "torque_leg": [],
        "torque_upper": [],
        "force": [],
        "pos_x": [],
        "pos_z": [], # Position du pied Z
        "hip_z": [], # <-- NOUVEAU : Position de la hanche Z
        "goal_x": [],
        "goal_z": [],
        "length": [],
        "length_des": []
    }
    
    sensor_id_contact = m.sensor('socle_contact').id
    hip_sensor_id = m.sensor('hip_position').adr
    
    for _ in range(steps):
        # Lecture capteurs
        hip_pos = float(d.sensordata[hip_sensor_id]) # Hauteur de la hanche
        
        adr = m.sensor_adr[sensor_id_contact]
        dim = m.sensor_dim[sensor_id_contact]
        foot_force = d.sensordata[adr:adr+dim]
        f_norm = np.linalg.norm(foot_force)
        
        # Appel contrôleur
        try:
            torque, pos, pos_des, L, Ld = control_func(m, d, -0.16, hip_pos, f_norm)
        except TypeError:
            torque, pos, pos_des, L, Ld = control_func(d, -0.16, hip_pos, f_norm)
        
        # Stockage des données
        data["torque_upper"].append(torque[0])
        data["torque_leg"].append(torque[1])
        data["force"].append(f_norm)
        data["pos_x"].append(pos[0])
        data["pos_z"].append(pos[1])
        data["hip_z"].append(hip_pos) # <-- NOUVEAU : On sauvegarde la hanche
        data["goal_x"].append(pos_des[0])
        data["goal_z"].append(pos_des[1])
        data["length"].append(L)
        data["length_des"].append(Ld)
        
        # Step simulation
        d.ctrl = torque
        mujoco.mj_step(m, d)
        
    return data

def plot_comparison(results_dict):
    colors = {"P": "red", "PD": "blue", "LQR": "green"}
    styles = {"P": "--", "PD": "-", "LQR": "-."}
    
    # --- FIGURE 1: POSITION Z DU PIED ---
    plt.figure(figsize=(10, 6))
    plt.title("Comparison: Foot Z Position (Relative to Hip)", fontsize=16)
    for label, data in results_dict.items():
        plt.plot(data["time"], data["pos_z"], label=label, color=colors[label], linestyle=styles[label], linewidth=2)
    plt.plot(results_dict["PD"]["time"], results_dict["PD"]["goal_z"], 'k', label="Goal (Swing)", linewidth=3, alpha=0.7)
    plt.xlabel("Time [s]",fontsize=15)
    plt.ylabel("Foot Position Z [m]",fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("./display/Compare_Foot_Z.png")
    
    # --- FIGURE 2: POSITION X DU PIED ---
    plt.figure(figsize=(10, 6))
    plt.title("Comparison: Foot X Tracking (Swing Stability)", fontsize=16)
    for label, data in results_dict.items():
        plt.plot(data["time"], data["pos_x"], label=label, color=colors[label], linestyle=styles[label], linewidth=2)
    plt.plot(results_dict["PD"]["time"], results_dict["PD"]["goal_x"], 'k', label="Goal", linewidth=3, alpha=0.7)
    plt.xlabel("Time [s]",fontsize=15)
    plt.ylabel("Foot Position X [m]",fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("./display/Compare_Foot_X.png")

    # --- FIGURE 3: TORQUES ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.set_title("Comparison: Torques", fontsize=16)
    for label, data in results_dict.items():
        ax1.plot(data["time"], data["torque_upper"], label=f"{label} Upper", color=colors[label], linewidth=2, linestyle=styles[label])
        ax2.plot(data["time"], data["torque_leg"], label=f"{label} Leg", color=colors[label],  linewidth=2,linestyle=styles[label])
    ax1.set_ylabel("Upper Torque [Nm]", fontsize=15)
    ax1.legend(loc="upper right",fontsize=15)
    ax2.set_ylabel("Leg Torque [Nm]", fontsize=15)
    ax2.set_xlabel("Time [s]", fontsize=15)
    ax2.legend(loc="upper right",fontsize=15)
    plt.tight_layout()
    plt.savefig("./display/Compare_Torques.png")

    # --- FIGURE 4: FORCE ---
    plt.figure(figsize=(10, 6))
    plt.title("Comparison: Ground Reaction Force", fontsize=16)
    for label, data in results_dict.items():
        plt.plot(data["time"], data["force"], label=label, color=colors[label], linestyle=styles[label], alpha=0.8, linewidth=2)
    plt.xlabel("Time [s]",fontsize=15)
    plt.ylabel("Force [N]",fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("./display/Compare_Forces.png")

    # --- FIGURE 5: LENGTH ---
    plt.figure(figsize=(10, 6))
    plt.title("Comparison: Virtual Leg Length (L)", fontsize=16)
    for label, data in results_dict.items():
        plt.plot(data["time"], data["length"], label=label, color=colors[label], linestyle=styles[label], linewidth=2)
    plt.plot(results_dict["PD"]["time"], results_dict["PD"]["length_des"], 'k', label="L_des (Goal)", linewidth=3, alpha=0.8)
    plt.xlabel("Time [s]",fontsize=15)
    plt.ylabel("Length [m]",fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("./display/Compare_Lengths.png")

    # --- FIGURE 6: HIP Z POSITION (NOUVEAU) ---
    plt.figure(figsize=(10, 6))
    plt.title("Comparison: Hip Absolute Vertical Position", fontsize=16)
    for label, data in results_dict.items():
        plt.plot(data["time"], data["hip_z"], label=label, color=colors[label], linestyle=styles[label], linewidth=2)
    
    plt.xlabel("Time [s]",fontsize=15)
    plt.ylabel("Hip Height Z [m] (World Frame)",fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("./display/Compare_Hip_Z.png")
    
    print("All 6 comparison plots saved in ./display/")

if __name__ == "__main__":
    model_file = "ref_stand.xml"
    sim_duration = 2.5
    
    all_results = {}
    
    # 1. PD
    controller.mx = 10.0; controller.mz = 10.0
    all_results["PD"] = run_simulation(model_file, controller.impedance_control, "PD", sim_time=sim_duration)
    
    # 2. P
    controller.mx = 0.0; controller.mz = 0.0
    all_results["P"] = run_simulation(model_file, controller.impedance_control, "P", sim_time=sim_duration)
    
    # 3. LQR
    all_results["LQR"] = run_simulation(model_file, controller_lqr.impedance_lqr, "LQR", sim_time=sim_duration)
    
    plot_comparison(all_results)
    plt.show()