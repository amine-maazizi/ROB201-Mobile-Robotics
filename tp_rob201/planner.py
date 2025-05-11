"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell):
        """
        Return the 8 neighboring cells of the current cell in the occupancy grid.
        current_cell: tuple (x, y) representing the cell's map coordinates
        Returns: list of tuples [(x1, y1), (x2, y2), ...] of valid neighboring cells
        """
        x, y = current_cell
        neighbors = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),  # Orthogonal neighbors
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)  # Diagonal neighbors
        ]
        valid_neighbors = []
        for nx, ny in neighbors:
            if (0 <= nx < self.grid.x_max_map and 
                0 <= ny < self.grid.y_max_map and 
                self.grid.occupancy_map[nx, ny] < 0.5):  # Check bounds and free space
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    def heuristic(self, cell1, cell2):
        """
        Compute the Euclidean distance between two cells in world coordinates.
        cell1, cell2: tuples (x, y) representing cell map coordinates
        Returns: float, Euclidean distance in meters
        """
        x1, y1 = cell1
        x2, y2 = cell2
        wx1, wy1 = self.grid.conv_map_to_world(x1, y1)
        wx2, wy2 = self.grid.conv_map_to_world(x2, y2)
        return np.sqrt((wx2 - wx1)**2 + (wy2 - wy1)**2)

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start: [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal: [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        Returns: list of [x, y, theta] poses in world coordinates forming the path
        """
        # Convert start and goal to map coordinates
        start_map = self.grid.conv_world_to_map(start[0], start[1])
        goal_map = self.grid.conv_world_to_map(goal[0], goal[1])

        # Initialize A* data structures
        open_set = [(0, start_map)]  # Priority queue: (f_score, cell)
        came_from = {}
        g_score = {start_map: 0}
        f_score = {start_map: self.heuristic(start_map, goal_map)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_map:
                # Reconstruct path
                path = []
                while current in came_from:
                    wx, wy = self.grid.conv_map_to_world(current[0], current[1])
                    path.append(np.array([wx, wy, 0]))  # Theta set to 0
                    current = came_from[current]
                wx, wy = self.grid.conv_map_to_world(start_map[0], start_map[1])
                path.append(np.array([wx, wy, 0]))
                return path[::-1]  # Reverse path to go from start to goal

            for neighbor in self.get_neighbors(current):
                # Distance to neighbor (1 for orthogonal, sqrt(2) for diagonal in map units)
                dist = 1.414 if abs(current[0] - neighbor[0]) + abs(current[1] - neighbor[1]) == 2 else 1
                tentative_g_score = g_score[current] + dist * self.grid.resolution

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_map)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # Return empty path if no path is found

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal