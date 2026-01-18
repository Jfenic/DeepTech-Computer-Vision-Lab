def quaternion_from_euler(pitch, yaw, roll):
    """Create a quaternion from Euler angles (in radians)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def rotate_cube_quaternion(points, pitch, yaw, roll):
    """Rotate the cube using quaternions from Euler angles (pitch, yaw, roll)."""
    q = quaternion_from_euler(pitch, yaw, roll)
    R = quaternion_to_rotation_matrix(q)
    return points @ R.T
"""
In this code practice: 3D ROTATION, the gimbal lock problem, and its solution.
"""
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting


def get_cube():
    """Generate the vertices of a cube centered at the origin."""
    r = [-1, 1]
    points = np.array([[x, y, z] for x in r for y in r for z in r])
    return points

def rotation_matrix_x(theta):
    """Rotation matrix around the X axis (pitch)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])
    
def rotation_matrix_y(theta):
    """Rotation matrix around the Y axis (yaw)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(theta):
  """Rotation matrix around the Z axis (roll)"""
  c, s = np.cos(theta), np.sin(theta)
  return np.array([
      [c, -s, 0],
      [s,  c, 0],
      [0,  0, 1]
  ])
  
    # Now we can define angle of rotation
def rotate_cube(points, pitch, yaw, roll):
    """Rotate the cube using Euler angles (pitch, yaw, roll)"""
    # Get rotation matrices for each axis
    R_x = rotation_matrix_x(pitch)
    R_y = rotation_matrix_y(yaw)
    R_z = rotation_matrix_z(roll)
    # Multiply the matrices in the correct order
    R = R_z @ R_y @ R_x
    return points @ R.T

def plot_cube(points, ax, title):
    """Plot the cube in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# Main execution
if __name__ == "__main__":
    cube = get_cube()
    plot_cube(cube, None, "Original Cube")
    # Normal rotation
    pitch = np.radians(45)  # Rotation around X
    yaw = np.radians(0)    # Rotation around Y
    roll = np.radians(0)   # Rotation around Z
    rotated_cube = rotate_cube(cube, pitch, yaw, roll)
    plot_cube(rotated_cube, None, "Rotated Cube using Euler Angles")

    # Gimbal Lock Example
    # Yaw = 90 degrees (π/2 radians), which aligns the X and Z axes
    pitch_gimbal = np.radians(0)
    yaw_gimbal = np.radians(90)
    roll_gimbal = np.radians(45)
    rotated_gimbal = rotate_cube(cube, pitch_gimbal, yaw_gimbal, roll_gimbal)
    plot_cube(rotated_gimbal, None, "Gimbal Lock: Yaw=90°, Roll=45° (Pitch=0°)")

    # Quaternion solution (no gimbal lock)
    rotated_quat = rotate_cube_quaternion(cube, pitch_gimbal, yaw_gimbal, roll_gimbal)
    plot_cube(rotated_quat, None, "Quaternion: Yaw=90°, Roll=45° (Pitch=0°)")