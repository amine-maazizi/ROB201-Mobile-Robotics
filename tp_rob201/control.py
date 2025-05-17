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
    import numpy as np

    # Parameters
    K_goal = 0.8  # Attractive gain
    K_obs = 2.0   # Repulsive gain (reduced for smoother avoidance)
    d_safe = 25   # Safe distance (slightly reduced)
    d_transition = 20  # Distance to switch to quadratic potential
    d_stop = 5    # Stopping distance

    # Current and goal positions
    q = current_pose[:2]
    q_goal = goal_pose[:2]
    d = np.linalg.norm(q_goal - q)

    # Attractive potential gradient
    if d < d_stop:
        attractive = np.zeros(2)  # Stop if close to goal
    elif d < d_transition:
        # Quadratic potential for smooth approach
        attractive = K_goal * (q_goal - q)  # Gradient: K_goal * (q_goal - q)
    else:
        # Linear potential
        attractive = K_goal * (q_goal - q) / d if d > 0 else np.zeros(2)

    # Repulsive potential gradient
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    repulsive = np.zeros(2)
    influence_range = d_safe * 1.5  # Consider obstacles within 1.5 * d_safe

    for dist, angle in zip(distances, angles):
        if dist < influence_range:
            # Obstacle position relative to robot
            q_obs = q + np.array([dist * np.cos(angle + current_pose[2]),
                                 dist * np.sin(angle + current_pose[2])])
            d_obs = dist - d_safe
            if d_obs < 0:
                # Repulsive force direction: away from obstacle
                repulsive += K_obs * (1/dist - 1/d_safe) * (q - q_obs) / (d_obs + 1e-6)

    # Total velocity
    V = attractive + repulsive  # Changed to + repulsive (direction is q - q_obs)

    # Linear and angular velocities
    V_linear = np.clip(np.linalg.norm(V), -0.6, 0.6)  # Slightly increased range
    theta = current_pose[2]
    V_angular = np.arctan2(V[1], V[0]) - theta
    V_angular = np.clip(V_angular, -0.8, 0.8)  # Reduced angular range for stability

    if verbose:
        print(f"Linear velocity: {V_linear}, Angular velocity: {V_angular}")
        print(f"Attractive: {attractive}, Repulsive: {repulsive}")
        print(f"Distance to goal: {d}, Min distance to obstacle: {np.min(distances)}")

    if debug_window is not None:
        debug_window.update_components({
            "attractive_vel": np.linalg.norm(attractive),
            "repulsive_vel": np.linalg.norm(repulsive),
            "d_obs": np.min(distances) - d_safe
        })

    command = {"forward": V_linear, "rotation": V_angular}
    return command


