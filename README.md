# 2-DOF Monopod Leg Simulation (ODRI)

This project simulates a 2-degree-of-freedom robotic leg based on the Open Dynamic Robot Initiative (ODRI) architecture using the **MuJoCo** physics engine and Python.

The simulation compares two control strategies for the swing phase during a jumping motion:
1.  **Impedance Control + Cartesian PD**
2.  **Impedance Control + Linear Quadratic Regulator (LQR)**

🎥 **Demo Videos:** Videos of the simulation for both controllers are available on the OneDrive repository

## 📂 Project Structure
Inside the folder * `Simulation/`
* `main.py`: The entry point of the simulation. Handles the simulation loop and data logging.
* `controller.py`: Contains the standard Impedance Controller with a PD controller for the swing phase.
* `controller_lqr.py`: Contains the Impedance Controller coupled with an LQR controller for the swing phase.
* `ref_stand.xml`: The MuJoCo model file defining the robot and environment.
* `display/`: Output folder where simulation plots are saved.

## 🚀 How to Run

1.  Ensure you have the required dependencies installed:
    ```bash
    pip install mujoco numpy matplotlib scipy
    ```

2.  Run the main script:
    ```bash
    python main.py
    ```

## ⚙️ Switching Controllers

To switch between the **Standard PD** controller and the **LQR** controller, you must modify `main.py`.

1.  Open `main.py`.
2.  Locate the control loop inside the `simulate_free_motion` function (around line 55).
3.  **Comment out** the controller you do not want to use and **uncomment** the one you wish to test.

**Example: Using the Standard PD Controller**
```python
# --- Standard Impedance Control (PD) ---
torque, pos, pos_des, L, Ld = impedance_control(d=d, x_hip=-0.16, z_hip=hip_pos, f_ground=f_norm)

# --- Impedance Control with LQR ---
# torque, pos, pos_des, L, Ld = impedance_lqr(m=m, d=d, x_hip=-0.16, z_hip=hip_pos, f_ground=f_norm)