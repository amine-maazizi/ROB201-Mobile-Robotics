""" A set of robotics control functions """

import random
import numpy as np
from math import fabs
from sklearn.cluster import DBSCAN


def reactive_obst_avoid(lidar, angle, target_angle, is_rotating):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """    
    safe_distance = 0.3
    max_speed = 0.5
    max_rotation = 1.0
    
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
    # Control parameters
    K_goal = 0.5
    K_obs = 5.0
    d_safe = 50
    d_transition = 80
    d_stop = 15
    eta = 0.3
    max_linear_vel = 0.3
    max_angular_vel = 1.0

    q = current_pose[:2]
    q_goal = goal_pose[:2]
    d = np.linalg.norm(q_goal - q)

    # Attractive potential with smooth transition near goal
    if d < d_stop:
        attractive = np.zeros(2)
        orientation_error = np.arctan2(np.sin(goal_pose[2] - current_pose[2]), 
                                     np.cos(goal_pose[2] - current_pose[2]))
    else:
        if d < d_transition:
            attractive = K_goal * (q_goal - q) * (d / d_transition)
        else:
            attractive = K_goal * (q_goal - q) / (d + 1e-6)
        orientation_error = 0

    # Repulsive potential from obstacles
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    repulsive = np.zeros(2)
    influence_range = d_safe * 2.0

    for dist, angle in zip(distances, angles):
        if dist < influence_range:
            q_obs = q + np.array([dist * np.cos(angle + current_pose[2]),
                                 dist * np.sin(angle + current_pose[2])])
            d_obs = dist - d_safe
            if d_obs < 0:
                repulsive += K_obs * (1/d_obs - 1/d_safe) * (q - q_obs) / (dist**2 + 1e-6)

    V = attractive - repulsive
    V_linear = np.clip(np.linalg.norm(V), 0, max_linear_vel)
    
    if d < d_stop:
        V_angular = eta * orientation_error
    else:
        desired_heading = np.arctan2(V[1], V[0])
        heading_error = np.arctan2(np.sin(desired_heading - current_pose[2]),
                                 np.cos(desired_heading - current_pose[2]))
        V_angular = eta * heading_error

    V_angular = np.clip(V_angular, -max_angular_vel, max_angular_vel)

    # Local minima escape
    if np.linalg.norm(V) < 0.01 and d > d_stop:
        V_angular += np.random.uniform(-0.2, 0.2)
        V_linear = 0.2

    if debug_window is not None:
        debug_window.update_components({
            "attractive_vel": np.linalg.norm(attractive),
            "repulsive_vel": np.linalg.norm(repulsive),
            "d_obs": np.min(distances) - d_safe
        })

    command = {"forward": V_linear, "rotation": V_angular}
    return command

def potential_field_control_w_clustering(lidar, current_pose, goal_pose, verbose=False, debug_window=None):

    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odometry frame
    goal_pose : [x, y, theta] nparray, target pose in odometry frame
    verbose : bool, whether to print debug information
    debug_window : DebugWindow, optional, for live updates
    Returns: command_dict
    """

    K_goal = 0.8  # Attractive gain
    K_obs = 2.0  # Increased repulsive gain for stronger avoidance
    d_safe = 25   # Reduced safe distance for earlier reaction
    d_transition = 50  # Increased for smoother transition
    d_stop = 10   # Tighter stopping distance



    cluster_eps = 10  # Distance max pour clustering (en unités de distance LIDAR)
    min_samples = 3   # Nombre min de points pour un cluster
    stuck_threshold = 0.05  # Seuil de vitesse pour détection blocage
    stuck_time = 3.0  # Temps max bloqué (s)

    # Current and goal positions
    q = current_pose[:2]
    q_goal = goal_pose[:2]
    d = np.linalg.norm(q_goal - q)

    # Attractive potential gradient
    if d < d_stop:
        attractive = np.zeros(2)  # Stop if close to goal
    elif d < d_transition:
        # Quadratic potential for smooth approach
        attractive = K_goal * (q_goal - q)
    else:
        # Linear potential
        attractive = K_goal * (q_goal - q) / d if d > 0 else np.zeros(2)

    # Repulsive potential gradient with obstacle segmentation
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    repulsive = np.zeros(2)
    influence_range = d_safe * 1.5  # Consider obstacles within 1.5 * d_safe

    # Segmentation des obstacles avec DBSCAN
    points = []
    valid_indices = np.where(distances < influence_range)[0]
    for i in valid_indices:
        dist = distances[i]
        angle = angles[i]
        x_obs = q[0] + dist * np.cos(angle + current_pose[2])
        y_obs = q[1] + dist * np.sin(angle + current_pose[2])
        points.append([x_obs, y_obs])

    if points:
        points = np.array(points)
        # Clustering avec DBSCAN
        clustering = DBSCAN(eps=cluster_eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # Calcul du gradient répulsif pour chaque cluster
        for label in set(labels) - {-1}:  # Exclure les points de bruit
            cluster_points = points[labels == label]
            # Centre du cluster (approximation de q_obs)
            q_obs = np.mean(cluster_points, axis=0)
            delta_obs = q_obs - q
            d_obs = np.linalg.norm(delta_obs) - d_safe
            if d_obs < 0 and d_obs > -d_safe:
                repulsive += K_obs * (1/(d_obs + d_safe) - 1/d_safe) * (q - q_obs) / (d_obs + 1e-6)

    # Total velocity
    V = attractive - repulsive

    # Linear and angular velocities
    V_linear = np.clip(np.linalg.norm(V), -0.6, 0.6)
    theta = current_pose[2]
    V_angular = np.arctan2(V[1], V[0]) - theta
    V_angular = np.arctan2(np.sin(V_angular), np.cos(V_angular))  # Normalisation
    V_angular = np.clip(V_angular, -0.8, 0.8)

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



def dynamic_window_control(lidar, current_pose, goal_pose, current_v, current_w, max_v=0.95, max_w=1.0, 
                          acc_v=0.5, acc_w=1.047, dt=0.25, alpha=0.8, beta=0.1, gamma=0.1, verbose=False, debug_window=None):
    """
    Implements the Dynamic Window Approach (DWA) for collision avoidance.
    Based on: Fox, D., Burgard, W., & Thrun, S. (1997). "The Dynamic Window Approach to Collision Avoidance."
    """
    # Velocity space discretization
    v_samples = np.linspace(0, max_v, 20)
    w_samples = np.linspace(-max_w, max_w, 20)

    # Dynamic window constraints
    v_range = [max(0, current_v - acc_v * dt), min(max_v, current_v + acc_v * dt)]
    w_range = [max(-max_w, current_w - acc_w * dt), min(max_w, current_w + acc_w * dt)]

    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    distances = np.array(distances)
    distances = np.where(distances <= 0, np.inf, distances)

    def compute_objective(v, w):
        # Predict trajectory
        if abs(w) < 1e-3:
            x_pred = current_pose[0] + v * np.cos(current_pose[2]) * dt
            y_pred = current_pose[1] + v * np.sin(current_pose[2]) * dt
            theta_pred = current_pose[2]
        else:
            R = v / w
            x_c = current_pose[0] - R * np.sin(current_pose[2])
            y_c = current_pose[1] + R * np.cos(current_pose[2])
            theta_pred = current_pose[2] + w * dt
            x_pred = x_c + R * np.sin(theta_pred)
            y_pred = y_c - R * np.cos(theta_pred)

        # Heading alignment score
        goal_vec = goal_pose[:2] - np.array([x_pred, y_pred])
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        heading = 180 - np.abs(np.degrees(goal_angle - theta_pred) % 360 - 180)
        heading = heading / 180.0

        # Obstacle clearance score
        dist = float('inf')
        for d, angle in zip(distances, angles):
            if np.isinf(d):
                continue
            obs_x = current_pose[0] + d * np.cos(angle + current_pose[2])
            obs_y = current_pose[1] + d * np.sin(angle + current_pose[2])
            
            if abs(w) < 1e-3:
                t = ((obs_x - current_pose[0]) * np.cos(current_pose[2]) +
                     (obs_y - current_pose[1]) * np.sin(current_pose[2])) / (v + 1e-6)
                t = np.clip(t, 0, dt)
                closest_x = current_pose[0] + v * t * np.cos(current_pose[2])
                closest_y = current_pose[1] + v * t * np.sin(current_pose[2])
            else:
                dist_to_center = np.sqrt((obs_x - x_c)**2 + (obs_y - y_c)**2)
                closest_dist = abs(dist_to_center - abs(R))
                obs_angle = np.arctan2(obs_y - y_c, obs_x - x_c)
                start_angle = current_pose[2] - np.pi/2 if w > 0 else current_pose[2] + np.pi/2
                end_angle = theta_pred - np.pi/2 if w > 0 else theta_pred + np.pi/2
                angle_in_range = (min(start_angle, end_angle) <= obs_angle <= max(start_angle, end_angle))
                if not angle_in_range:
                    closest_dist = np.sqrt((obs_x - x_pred)**2 + (obs_y - y_pred)**2)
                closest_x, closest_y = obs_x, obs_y
            d_traj = np.sqrt((obs_x - closest_x)**2 + (obs_y - closest_y)**2)
            dist = min(dist, d_traj)

        # Safety check
        dist = max(dist, 0.1)
        v_stop = np.sqrt(2 * dist * acc_v) if dist > 0 else 0
        w_stop = np.sqrt(2 * dist * acc_w) if dist > 0 else 0

        if v > v_stop or abs(w) > w_stop:
            return -float('inf')

        dist_norm = min(dist / 5.0, 1.0)
        vel_norm = v / max_v
        score = alpha * heading + beta * dist_norm + gamma * vel_norm
        return score

    # Find optimal velocity
    best_score = -float('inf')
    best_v, best_w = 0.0, 0.0

    for v in v_samples:
        if v < v_range[0] or v > v_range[1]:
            continue
        for w in w_samples:
            if w < w_range[0] or w > w_range[1]:
                continue
            score = compute_objective(v, w)
            if score > best_score:
                best_score = score
                best_v, best_w = v, w

    command = {"forward": best_v, "rotation": best_w}

    if debug_window is not None:
        goal_vec = goal_pose[:2] - current_pose[:2]
        attractive_vel = np.linalg.norm(goal_vec) * alpha
        min_dist = np.min(distances)
        repulsive_vel = beta * (1.0 - min(min_dist / 5.0, 1.0))
        
        debug_window.update_components({
            "attractive_vel": attractive_vel,
            "repulsive_vel": repulsive_vel,
            "d_obs": min_dist
        })

    return command