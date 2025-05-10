""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0], dtype=np.float64)

    def _score(self, lidar, pose, method="direct"):
        """
        Computes the score of laser end points in the map using either direct summation or bilinear interpolation.
        lidar: placebot object with lidar data
        pose: [x, y, theta] nparray, position of the robot in world coordinates
        method: str, "direct" for direct summation, "bilinear" for bilinear interpolation
        """
        MAX_RANGE = lidar.max_range
        lidar_values, lidar_angles = lidar.get_sensor_values(), lidar.get_ray_angles()
        lidar_angles = lidar_angles[lidar_values < MAX_RANGE]    
        lidar_values = lidar_values[lidar_values < MAX_RANGE]

        # Lidar -> world coordinates
        detected_points = TinySlam.pol_to_cart2(lidar_values, lidar_angles, pose)
        x_world, y_world = detected_points[0], detected_points[1]

        #  world -> map coordinates
        map_points = self.grid.conv_world_to_map(x_world, y_world)
        x_indices_float, y_indices_float = map_points[0], map_points[1]

        total_score = 0.0

        if method == "direct":
            # Sum of the values in the grid
            x_indices = np.clip(x_indices_float, 0, self.grid.x_max_map - 1).astype(int)
            y_indices = np.clip(y_indices_float, 0, self.grid.y_max_map - 1).astype(int)
            total_score = np.sum(self.grid.occupancy_map[x_indices, y_indices])

        elif method == "bilinear":
            # Bilinear interpolation method
            for x, y, x_idx, y_idx in zip(x_world, y_world, x_indices_float, y_indices_float):
                # Skip points outside the map
                if x_idx < 0 or x_idx >= self.grid.x_max_map - 1 or y_idx < 0 or y_idx >= self.grid.y_max_map - 1:
                    continue

                # Find the four surrounding grid points
                x0_idx = int(x_idx)
                y0_idx = int(y_idx)
                x1_idx = x0_idx + 1
                y1_idx = y0_idx + 1

                # Convert grid indices back to world coordinates to get x0, x1, y0, y1
                x0, y0 = self.grid.conv_map_to_world(x0_idx, y0_idx)
                x1, y1 = self.grid.conv_map_to_world(x1_idx, y1_idx)

                # Get occupancy values at the four corners (M(P_00), M(P_01), M(P_10), M(P_11))
                m_00 = self.grid.occupancy_map[x0_idx, y0_idx]
                m_01 = self.grid.occupancy_map[x0_idx, y1_idx]
                m_10 = self.grid.occupancy_map[x1_idx, y0_idx]
                m_11 = self.grid.occupancy_map[x1_idx, y1_idx]

                # Compute interpolation weights
                dx = (x - x0) / (x1 - x0) if x1 != x0 else 0
                dy = (y - y0) / (y1 - y0) if y1 != y0 else 0

                # Bilinear interpolation formula
                score = (1 - dy) * ((1 - dx) * m_00 + dx * m_10) + dy * ((1 - dx) * m_01 + dx * m_11)
                total_score += score

        else:
            raise ValueError("Method must be 'direct' or 'bilinear'")

        return total_score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        
        x0, y0, theta0 = odom_pose_ref
        x, y, theta = odom_pose

        corrected_pose = np.array([x0 + x * np.cos(theta0) - y * np.sin(theta0),
                                  y0 + x * np.sin(theta0) + y * np.cos(theta0),
                                  theta0 + theta])
        

        return corrected_pose

    def localise(self, lidar, raw_odom_pose, N=200, variance=0.2):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """

        # Initial score with current odom_pose_ref
        initial_absolute_pose = self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref)
        best_score = self._score(lidar, initial_absolute_pose, method="bilinear")
        best_odom_pose_ref = self.odom_pose_ref.copy()  
        no_improvement = 0

        while no_improvement < N:
            offset = np.random.normal(0, variance, 3)
            new_pose_ref = best_odom_pose_ref + offset
            new_absolute_pose = self.get_corrected_pose(raw_odom_pose, new_pose_ref)
            score = self._score(lidar, new_absolute_pose)

            if score > best_score:
                best_score = score
                best_odom_pose_ref = new_pose_ref
                no_improvement = 0
            else:
                no_improvement += 1

        if best_score > 7000:
            self.odom_pose_ref = best_odom_pose_ref

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # Get the corrected pose using the current odom_pose_ref
        x0, y0, _ = pose
        
        # Convert lidar measurements to world coordinates
        detected_points = TinySlam.pol_to_cart2(lidar.get_sensor_values(), lidar.get_ray_angles(), pose)
        
        PADDING = 20
        for x, y in zip(*detected_points):
            target_x, target_y = x, y
            if x0 != x:
                target_x += (PADDING) if x0 > x else (-PADDING)
            if y0 != y:
                target_y += (PADDING) if y0 > y else (-PADDING)

            self.grid.add_value_along_line(x0, y0, target_x, target_y, -1.99)

        self.grid.add_map_points(detected_points[0], detected_points[1], 3.98)
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)

    @staticmethod
    def pol_to_cart2(ranges, ray_angles, pose):
        x, y, theta = pose
        pts_x = ranges * np.cos(ray_angles + theta) + x
        pts_y = ranges * np.sin(ray_angles + theta) + y
        return np.vstack([pts_x, pts_y]) 
