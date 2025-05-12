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

def potential_field_control(lidar, current_pose, goal_pose, verbose=False, debug_window=None):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odometry frame
    goal_pose : [x, y, theta] nparray, target pose in odometry frame
    verbose : bool, whether to print debug information
    debug_window : DebugWindow, optional, for live updates
    Returns: command_dict
    """
    
    # Attractive potential gradient for goal
    K_goal = 0.8
    q = current_pose[:2]
    q_goal = goal_pose[:2]

    d = np.linalg.norm(q_goal - q)
    if d > 0:
        attractive = K_goal * (q_goal - q) / d
    else:
        attractive = np.zeros(2)
    
    # Repulsive potential gradient for obstacles
    K_obs = 5.0
    d_safe = 30
    distances = lidar.get_sensor_values()
    min_dist = np.min(distances)
    d_obs = min_dist - d_safe
    repulsive = np.zeros(2)
    if d_obs < 0:
        repulsive = K_obs * (1/min_dist - 1/d_safe) * (q - q_goal) / d_obs
    
    # Total velocity
    V = attractive - repulsive
    
    # Linear and angular velocities
    V_linear = np.clip(np.linalg.norm(V), -0.5, 0.5)
    theta = current_pose[2]
    V_angular = np.arctan2(V[1], V[0]) - theta
    V_angular = np.clip(V_angular, -1.0, 1.0)
    
    if verbose:
        print(f"Linear velocity: {V_linear}, Angular velocity: {V_angular}")
        print(f"Attractive: {attractive}, Repulsive: {repulsive}")
    
    if debug_window is not None:
        debug_window.update_components({
            "attractive_vel": np.linalg.norm(attractive),
            "repulsive_vel": np.linalg.norm(repulsive),
            "d_obs": d_obs
        })
    command = {"forward": V_linear, "rotation": V_angular}
    return command


