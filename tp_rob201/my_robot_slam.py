"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam
from debug_window import DebugWindow

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # TP1
        self.target_angle = np.random.rand() * (2 * np.pi) - np.pi
        self.is_rotating = False

        # TP2
        self.robot_position = np.array(robot_position)
        self.goal = np.array([0, 0])

        # TP4
        self.iteration = 0
        self.score_threshold = 7000

        # Debug window
        self.debug_window = DebugWindow()
        self.last_command = {"forward": 0.0, "rotation": 0.0}

    def select_new_goal(self):
        lidar = self.lidar()
        laser_dist = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        
        obstacle_rays = np.where(laser_dist < lidar.max_range)[0]
        non_colliding_rays = np.where(laser_dist >= lidar.max_range)[0]
        
        if len(obstacle_rays) > 0:
            goal_ray = np.random.choice(obstacle_rays)
            ray_dist = laser_dist[goal_ray]
            D = 50
            goal_dist = max(ray_dist - D, 10)
        else:
            goal_ray = np.random.choice(non_colliding_rays)
            goal_dist = lidar.max_range * 0.8
        
        ray_angle = ray_angles[goal_ray]
        pose = self.odometer_values()
        world_pose = np.array([self.robot_position[0] + pose[0], self.robot_position[1] + pose[1], pose[2]])
        
        goal_x = world_pose[0] + goal_dist * np.cos(ray_angle + world_pose[2])
        goal_y = world_pose[1] + goal_dist * np.sin(ray_angle + world_pose[2])
        
        return np.array([goal_x, goal_y]) - self.robot_position

    def control(self):
        """
        Main control function executed at each time step
        """

        return self.control_tp4()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        # Compute new command speed to perform obstacle avoidance
        _, _, theta = self.odometer_values()
        command, self.is_rotating = reactive_obst_avoid(self.lidar(), theta, self.target_angle, self.is_rotating)

        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, exploration using lidar data and path planning
        """
        pose = self.odometer_values()  

        world_pose = np.array([self.robot_position[0] + pose[0], self.robot_position[1] + pose[1], pose[2]])
        goal_in_odom = self.goal + pose[:2] - world_pose[:2] 

        dist = np.linalg.norm(self.goal - world_pose[:2])  
        if dist <= 10:
            if self.iteration >= 25:
                new_goal = self.select_new_goal()
                self.goal = new_goal
                self.debug_window.add_status_message(
                    f"New goal set to: ({new_goal[0]:.1f}, {new_goal[1]:.1f})",
                    self.debug_window.success_color
                )
            goal_in_odom = self.goal
         
        self.iteration += 1
        command = potential_field_control(self.lidar(), pose, np.append(goal_in_odom, pose[2]))
        return command

    def control_tp3(self):
        pose = self.odometer_values()
        self.tiny_slam.update_map(self.lidar(), pose)
        self.tiny_slam.grid.display_cv(pose)

        return self.control_tp2()
    
    def control_tp4(self):
        """
        Control function for TP4 with SLAM 
        """
        raw_odom = self.odometer_values()
        
        score = self.tiny_slam.localise(self.lidar(), raw_odom)
        
        corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        if score > self.score_threshold:
            old_position = self.robot_position.copy()
            self.robot_position = corrected_pose[:2]
            self.debug_window.add_status_message(
                f"Position updated: ({old_position[0]:.1f}, {old_position[1]:.1f}) → ({self.robot_position[0]:.1f}, {self.robot_position[1]:.1f})",
                self.debug_window.warning_color
            )
      
        self.tiny_slam.update_map(self.lidar(), corrected_pose)
        # self.tiny_slam.grid.display_cv(corrected_pose)
        
        # Get command and store it for debug window
        command = self.control_tp2()
        self.last_command = command
        
        # Calculate world position for debug window
        world_pose = np.array([
            self.robot_position[0] + corrected_pose[0],
            self.robot_position[1] + corrected_pose[1]
        ])
        
        # Update and render debug window with explicit values
        self.debug_window.update(
            position=world_pose,
            goal=self.goal,
            speed=command["forward"],
            rotation=command["rotation"],
            slam_score=score,
            iteration=self.iteration,
            max_range=self.lidar().max_range
        )
        self.debug_window.render()
        
        return command  