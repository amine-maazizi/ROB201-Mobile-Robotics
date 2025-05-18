""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
import cma
from occupancy_grid import OccupancyGrid

class TinySlam:
    """Simple occupancy grid SLAM with optimized point reduction"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0], dtype=np.float64)
        
        self.score_threshold = 4000

    def _score(self, lidar, pose, method="direct"):
        """
        Computes the score of laser end points in the map using either direct summation or bilinear interpolation.
        lidar: placebot object with lidar data
        pose: [x, y, theta] nparray, position of the robot in world coordinates
        method: str, "direct" for direct summation, "bilinear" for bilinear interpolation
        """
        import numpy as np

        MAX_RANGE = lidar.max_range
        lidar_values, lidar_angles = lidar.get_sensor_values(), lidar.get_ray_angles()
        lidar_angles = lidar_angles[lidar_values < MAX_RANGE]    
        lidar_values = lidar_values[lidar_values < MAX_RANGE]

        # Lidar -> world coordinates
        detected_points = TinySlam.pol_to_cart2(lidar_values, lidar_angles, pose)
        x_world, y_world = detected_points[0], detected_points[1]

        # World -> map coordinates
        map_points = self.grid.conv_world_to_map(x_world, y_world)
        x_indices_float, y_indices_float = map_points[0], map_points[1]

        total_score = 0.0

        def bilinear_interpolate(x, y, x_idx, y_idx):
            """Internal function to compute bilinear interpolation for a single point."""
            # Skip points outside the map
            if x_idx < 0 or x_idx >= self.grid.x_max_map - 1 or y_idx < 0 or y_idx >= self.grid.y_max_map - 1:
                return 0.0

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
            return (1 - dy) * ((1 - dx) * m_00 + dx * m_10) + dy * ((1 - dx) * m_01 + dx * m_11)

        if method == "direct":
            # Sum of the values in the grid
            x_indices = np.clip(x_indices_float, 0, self.grid.x_max_map - 1).astype(int)
            y_indices = np.clip(y_indices_float, 0, self.grid.y_max_map - 1).astype(int)
            total_score = np.sum(self.grid.occupancy_map[x_indices, y_indices])

        elif method == "bilinear":
            # Use the internal bilinear interpolation function for each point
            scores = np.array([
                bilinear_interpolate(x, y, x_idx, y_idx)
                for x, y, x_idx, y_idx in zip(x_world, y_world, x_indices_float, y_indices_float)
            ])
            total_score = np.sum(scores)

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

    def localise(self, lidar, raw_odom_pose, N=100, variance=0.3, localisation_method="simple"):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        N : int, number of iterations or population size
        variance : float, initial variance for random sampling
        localisation_method : str, method to use ("simple", "cem", "cma-es")
        """
        # Initial pose and score
        initial_absolute_pose = self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref)
        best_score = self._score(lidar, initial_absolute_pose, method="direct")
        best_odom_pose_ref = self.odom_pose_ref.copy()

        def run_optimisation(method):
            """Internal function to run localisation optimisation."""
            nonlocal best_score, best_odom_pose_ref

            if method == "simple":
                no_improvement = 0
                while no_improvement < N:
                    offset = np.random.normal(0, variance, 3)
                    new_pose_ref = best_odom_pose_ref + offset
                    new_absolute_pose = self.get_corrected_pose(raw_odom_pose, new_pose_ref)
                    score = self._score(lidar, new_absolute_pose)

                    if score > best_score:
                        best_score = score
                        best_odom_pose_ref = new_pose_ref.copy()
                        no_improvement = 0
                    else:
                        no_improvement += 1

            elif method == "cem":
                mu = best_odom_pose_ref.copy()
                sigma = variance
                elite_fraction = 0.1
                for _ in range(N // 10):  # Run for fewer iterations with population
                    population = np.random.normal(mu, sigma, (10, 3))
                    scores = np.array([
                        self._score(lidar, self.get_corrected_pose(raw_odom_pose, pose))
                        for pose in population
                    ])
                    elite_idx = np.argsort(scores)[-int(10 * elite_fraction):]
                    elite_poses = population[elite_idx]
                    mu = np.mean(elite_poses, axis=0)
                    sigma = np.std(elite_poses, axis=0).mean() + 1e-3  # Add small epsilon
                    if scores[elite_idx[-1]] > best_score:
                        best_score = scores[elite_idx[-1]]
                        best_odom_pose_ref = population[elite_idx[-1]].copy()

            elif method == "cma-es":
                if cma is None:
                    raise ImportError("CMA-ES requires the 'cma' package. Install it via pip.")
                options = {'maxiter': N // 10, 'popsize': 10, 'verbose': -1}
                result = cma.fmin(
                    lambda x: -self._score(lidar, self.get_corrected_pose(raw_odom_pose, x)),
                    best_odom_pose_ref.copy(), variance, options=options
                )
                if -result[1] > best_score:
                    best_score = -result[1]
                    best_odom_pose_ref = result[0].copy()

            else:
                raise ValueError("Unknown localisation method")

        # Run the selected optimisation method
        run_optimisation(localisation_method)

        # Update odometry reference if score exceeds threshold
        if best_score > self.score_threshold:
            self.odom_pose_ref = best_odom_pose_ref

        return best_score
    
    def filter_points(self, x_points, y_points, method="regular", stride=2, min_distance=None):
        """
        Filter lidar points to reduce computation
        
        x_points, y_points: coordinates of detected points
        method: "regular" to keep 1 point every stride points
                "adaptive" to keep points with minimum distance between them
        stride: for regular method, keep 1 point every stride points
        min_distance: for adaptive method, minimum distance between points (map units)
        
        Returns: filtered x and y coordinates
        """
        
        if len(x_points) == 0:
            return x_points, y_points
        
        if method == "regular":
            # Simple regular sampling - take every n-th point
            indices = np.arange(0, len(x_points), stride)
            return x_points[indices], y_points[indices]
        
        elif method == "adaptive":
            if min_distance is None:
                # Default to grid cell size if not specified
                min_distance = self.grid.resolution
                
            # Calculate cell indices for all points
            map_points = self.grid.conv_world_to_map(x_points, y_points)
            cell_x, cell_y = np.floor(map_points[0]).astype(int), np.floor(map_points[1]).astype(int)
            
            # Create a unique cell identifier for each point
            cell_ids = cell_x * self.grid.y_max_map + cell_y
            
            # Get unique cell IDs and their first occurrences
            unique_cells, indices = np.unique(cell_ids, return_index=True)
            
            return x_points[indices], y_points[indices]
        
        else:
            raise ValueError("Method must be 'regular' or 'adaptive'")
    
    def update_map(self, lidar, pose, sensor_model="simple", point_reduction="none", stride=2, min_distance=None):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        sensor_model : str, sensor model to use ("simple", "intermediate", "gaussian", "noisy")
        point_reduction: str, method to reduce points ("none", "regular", "adaptive")
        stride: int, for regular reduction, keep 1 point every stride points
        min_distance: float, for adaptive reduction, minimum distance between points
        """
        import numpy as np

        # Get the corrected pose
        x0, y0, _ = pose
        
        # Convert lidar measurements to world coordinates
        lidar_values, lidar_angles = lidar.get_sensor_values(), lidar.get_ray_angles()
        
        # Filter for valid range values
        MAX_RANGE = lidar.max_range
        valid_indices = lidar_values < MAX_RANGE
        lidar_values = lidar_values[valid_indices]
        lidar_angles = lidar_angles[valid_indices]
        
        # Convert to world coordinates
        detected_points = TinySlam.pol_to_cart2(lidar_values, lidar_angles, pose)
        x_world, y_world = detected_points[0], detected_points[1]
        
        # Apply point reduction if requested
        if point_reduction != "none":
            x_world, y_world = self.filter_points(
                x_world, y_world, 
                method=point_reduction, 
                stride=stride, 
                min_distance=min_distance
            )
            # Reconstitute detected_points after filtering
            detected_points = np.vstack([x_world, y_world])
        
        def get_probabilities(model, dist, max_dist, is_occupied):
            """Calculate probability updates based on sensor model."""
            if model == "simple":
                return 3.98 if is_occupied else -1.99
            elif model == "intermediate":
                factor = 1 - dist / max_dist if not is_occupied else dist / max_dist
                return (3.98 if is_occupied else -1.99) * max(0.1, factor)
            elif model == "gaussian":
                sigma = 10.0  # Standard deviation for Gaussian
                prob = np.exp(-0.5 * (dist / sigma) ** 2)
                return (3.98 if is_occupied else -1.99) * prob
            elif model == "noisy":
                noise = np.random.normal(0, 0.2)  # Small Gaussian noise
                return (3.98 if is_occupied else -1.99) + noise
            else:
                raise ValueError("Unknown sensor model")

        PADDING = 20
        max_range = lidar.max_range
        for x, y in zip(x_world, y_world):
            target_x, target_y = x, y
            if x0 != x:
                target_x += (PADDING) if x0 > x else (-PADDING)
            if y0 != y:
                target_y += (PADDING) if y0 > y else (-PADDING)

            # Update probabilities along the line (free space)
            self.grid.add_value_along_line(
                x0, y0, target_x, target_y, 
                get_probabilities(sensor_model, 0, max_range, is_occupied=False)
            )

        # Update probabilities at detected points (occupied)
        self.grid.add_map_points(
            x_world, y_world, 
            get_probabilities(sensor_model, 0, max_range, is_occupied=True)
        )
        
        # Clip occupancy map to prevent extreme values
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)

    @staticmethod
    def pol_to_cart2(ranges, ray_angles, pose):
        x, y, theta = pose
        pts_x = ranges * np.cos(ray_angles + theta) + x
        pts_y = ranges * np.sin(ray_angles + theta) + y
        return np.vstack([pts_x, pts_y]) 