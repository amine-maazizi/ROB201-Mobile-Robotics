""" A set of robotics control functions """

import random
import numpy as np
from math import fabs

def reactive_obst_avoid(lidar, angle, target_angle, is_rotating):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """    
    # Safe distance from wall
    safe_distance = 0.3
    max_speed = 0.5
    max_rotation = 1.0
    
    # Get lidar distances
    distances = lidar.get_sensor_values()
    min_distance = np.min(distances)
    
    if min_distance > safe_distance and not is_rotating:
        speed = max_speed
        rotation_speed = 0.0
    else:
        speed = 0.0
        rotation_speed = max_rotation if angle < target_angle else -max_rotation
    
    command = {"forward": speed,
               "rotation": rotation_speed}
    
    return command

def potential_field_control(lidar, current_pose, goal_pose, verbose=False):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odometry frame
    goal_pose : [x, y, theta] nparray, target pose in odometry frame
    verbose : bool, whether to print debug information
    """
    
    # Attractive potential gradient for goal
    K_goal = 1.0
    q = current_pose
    q_goal = goal_pose

    d = np.linalg.norm(q_goal - q)
    if d > 0:
        V = K_goal * (q_goal - q) / d
    else:
        V = np.zeros(2)
    
    # Repulsive potential gradient for obstacles
    K_obs = 0.5
    d_safe = 0.5
    distances = lidar.get_sensor_values()
    min_dist = np.min(distances)
    d_obs = min_dist - d_safe
    if min_dist < d_safe and d_obs > 0:
        V += K_obs * (1/min_dist - 1/d_safe) * (q - q_goal) / d_obs
    
    # Linear and angular velocities
    V_linear = np.clip(np.linalg.norm(V), -0.5, 0.5)
    V_angular = np.arctan2(V[1], V[0]) - current_pose[2]
    V_angular = np.clip(V_angular, -1.0, 1.0)
    
    if verbose:
        print(f"Linear velocity: {V_linear}, Angular velocity: {V_angular}")
    
    return {"forward": V_linear, "rotation": V_angular}


