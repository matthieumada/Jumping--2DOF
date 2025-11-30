import numpy as np 
from scipy.spatial.transform import Rotation as R 

# This file is a converter from rotation angle in degree with a precise axis (x, y or z) 
# to quaternionin the format (w, x, y, z)used by Mujoco

def main(angle, axis):
    angle = np.deg2rad(angle)  # Convert degrees to radians
    rot = R.from_euler(axis, angle)
    quat_xyzw = rot.as_quat()  # Returns in the order (x, y, z, w)
    quat_wxyz = np.roll(quat_xyzw, 1)  # Roll to (w, x, y, z) for Mujoco
    return quat_wxyz


if __name__ == "__main__":
    # Input rotation angle and axis
    angle = float(input("Enter rotation angle in degrees (°): "))
    axis = input("Enter rotation axis (x, y, or z): ").lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    # Compute the quaternion
    quat_a = main(angle,axis)
    # display the value 
    print("Quaternion:", quat_a)
    quat_b = main(45,'x')
    
    if angle == 180 :
        quat_a = main(90, axis)
        print("new rotation  of 180 quaternion:",quat_a*quat_a)
        print("new quaternion:", quat_a)
