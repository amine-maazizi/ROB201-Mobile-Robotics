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

    # Parameters
    K_goal = 0.8  # Attractive gain
    K_obs = 1.0  # Repulsive gain (reduced for smoother avoidance)
    d_safe = 30   # Safe distance (slightly reduced)
    d_transition = 40  # Distance to switch to quadratic potential
    d_stop = 20    # Stopping distance

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
    V = attractive - repulsive 

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
    # Parameters
    K_goal = 0.8  # Attractive gain
    K_obs = 30.0  # Repulsive gain (reduced for smoother avoidance)
    d_safe = 30   # Safe distance (slightly reduced)
    d_transition = 40  # Distance to switch to quadratic potential
    d_stop = 20    # Stopping distance

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



def dynamic_window_control(lidar, current_pose, goal_pose, max_v=0.95, max_w=1.0, 
                          acc_v=0.5, acc_w=1.047, dt=0.25, alpha=0.8, beta=0.1, gamma=0.1, verbose=False):
    """
    Implements the Dynamic Window Approach (DWA) for collision avoidance, as described in:
    Fox, D., Burgard, W., & Thrun, S. (1997). "The Dynamic Window Approach to Collision Avoidance."
    IEEE Robotics & Automation Magazine, March 1997.
    
    This method selects optimal translational and rotational velocities by:
    1. Defining a search space of velocities based on the robot's dynamics.
    2. Restricting to admissible velocities (safe stopping) and the dynamic window (reachable velocities).
    3. Maximizing an objective function balancing goal heading, obstacle clearance, and speed.
    
    Args:
        lidar: placebot object with lidar data
        current_pose: [x, y, theta] nparray, current pose in odometry frame (m, m, rad)
        goal_pose: [x, y, theta] nparray, target pose in odometry frame (m, m, rad)
        max_v: float, maximum translational velocity (m/s), default 0.95 as per paper
        max_w: float, maximum rotational velocity (rad/s), default 1.0 (~57 deg/s)
        acc_v: float, maximum translational acceleration (m/s^2), default 0.5 as per paper
        acc_w: float, maximum rotational acceleration (rad/s^2), default 1.047 (~60 deg/s^2)
        dt: float, time interval for velocity updates (s), default 0.25 as per paper
        alpha: float, weight for heading in objective function, default 0.8 as per paper
        beta: float, weight for obstacle clearance, default 0.1 as per paper
        gamma: float, weight for velocity, default 0.1 as per paper
        verbose: bool, whether to print debug information
    
    Returns:
        command_dict: {"forward": v, "rotation": w}, where v is linear velocity (m/s) and w is angular velocity (rad/s)
    """
    # Current velocity (assumed measurable, e.g., from wheel encoders)
    # For simplicity, assume starting from rest if not provided; in practice, this should be updated
    current_v = 0.0
    current_w = 0.0

    # Step 1: Define the search space
    # Discretize the velocity space (v, w)
    v_samples = np.linspace(0, max_v, 20)  # 0 to max_v
    w_samples = np.linspace(-max_w, max_w, 20)  # -max_w to max_w

    # Step 2: Restrict to dynamic window (reachable velocities)
    v_range = [max(0, current_v - acc_v * dt), min(max_v, current_v + acc_v * dt)]
    w_range = [max(-max_w, current_w - acc_w * dt), min(max_w, current_w + acc_w * dt)]

    # Step 3: Get lidar data for obstacle detection
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # Validate lidar data
    distances = np.array(distances)
    distances = np.where(distances <= 0, np.inf, distances)  # Replace invalid distances with infinity

    if verbose:
        print(f"Dynamic window: v_range={v_range}, w_range={w_range}")
        print(f"Min lidar distance: {np.min(distances)}")

    def compute_objective(v, w):
        """Internal function to compute the objective function for a velocity pair (v, w)."""
        # Predict the trajectory (circular arc) over the time interval dt
        if abs(w) < 1e-3:  # Treat very small w as straight line to avoid division by zero
            # Straight line
            x_pred = current_pose[0] + v * np.cos(current_pose[2]) * dt
            y_pred = current_pose[1] + v * np.sin(current_pose[2]) * dt
            theta_pred = current_pose[2]
        else:
            # Circular arc
            R = v / w  # Radius of curvature
            x_c = current_pose[0] - R * np.sin(current_pose[2])  # Center of circle
            y_c = current_pose[1] + R * np.cos(current_pose[2])
            theta_pred = current_pose[2] + w * dt
            x_pred = x_c + R * np.sin(theta_pred)
            y_pred = y_c - R * np.cos(theta_pred)

        # Heading: Alignment with goal
        goal_vec = goal_pose[:2] - np.array([x_pred, y_pred])
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        heading = 180 - np.abs(np.degrees(goal_angle - theta_pred) % 360 - 180)  # 180 is best, 0 is worst
        heading = heading / 180.0  # Normalize to [0, 1]

        # Distance to closest obstacle
        dist = float('inf')
        for d, angle in zip(distances, angles):
            if np.isinf(d):  # Skip infinite distances
                continue
            # Obstacle position in world coordinates
            obs_x = current_pose[0] + d * np.cos(angle + current_pose[2])
            obs_y = current_pose[1] + d * np.sin(angle + current_pose[2])
            # Distance from obstacle to trajectory
            if abs(w) < 1e-3:
                # Straight line: closest point on line
                t = ((obs_x - current_pose[0]) * np.cos(current_pose[2]) +
                     (obs_y - current_pose[1]) * np.sin(current_pose[2])) / (v + 1e-6)
                t = np.clip(t, 0, dt)
                closest_x = current_pose[0] + v * t * np.cos(current_pose[2])
                closest_y = current_pose[1] + v * t * np.sin(current_pose[2])
            else:
                # Circular arc: distance to circle center
                dist_to_center = np.sqrt((obs_x - x_c)**2 + (obs_y - y_c)**2)
                closest_dist = abs(dist_to_center - abs(R))
                # Check if the obstacle is within the arc's angular range
                obs_angle = np.arctan2(obs_y - y_c, obs_x - x_c)
                start_angle = current_pose[2] - np.pi/2 if w > 0 else current_pose[2] + np.pi/2
                end_angle = theta_pred - np.pi/2 if w > 0 else theta_pred + np.pi/2
                angle_in_range = (min(start_angle, end_angle) <= obs_angle <= max(start_angle, end_angle))
                if not angle_in_range:
                    closest_dist = np.sqrt((obs_x - x_pred)**2 + (obs_y - y_pred)**2)
                closest_x, closest_y = obs_x, obs_y
            d_traj = np.sqrt((obs_x - closest_x)**2 + (obs_y - closest_y)**2)
            dist = min(dist, d_traj)

        # Admissible velocity check: can the robot stop before hitting the obstacle?
        # Add a minimum distance threshold to avoid overly restrictive stopping conditions
        dist = max(dist, 0.1)  # Minimum distance of 0.1m to prevent zero stopping velocity
        v_stop = np.sqrt(2 * dist * acc_v) if dist > 0 else 0
        w_stop = np.sqrt(2 * dist * acc_w) if dist > 0 else 0

        if verbose:
            print(f"v={v:.3f}, w={w:.3f}, dist={dist:.3f}, v_stop={v_stop:.3f}, w_stop={w_stop:.3f}")

        if v > v_stop or abs(w) > w_stop:
            return -float('inf')  # Not admissible

        # Normalize distance (assume max distance is 5m for normalization)
        dist_norm = min(dist / 5.0, 1.0)

        # Velocity: Prefer higher translational speed
        vel_norm = v / max_v

        # Objective function: weighted sum
        score = alpha * heading + beta * dist_norm + gamma * vel_norm
        return score

    # Step 4: Search for the best velocity in the dynamic window
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

    if verbose:
        print(f"Best score: {best_score}, Best v: {best_v}, Best w: {best_w}")

    # Step 5: Return the command
    command = {"forward": best_v, "rotation": best_w}
    return command