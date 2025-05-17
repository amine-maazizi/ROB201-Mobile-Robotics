import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam
from debug_window import DebugWindow

from control import potential_field_control, reactive_obst_avoid, potential_field_control_w_clustering, dynamic_window_control
from occupancy_grid import OccupancyGrid
from planner import Planner


class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # Step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # Storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # TP1
        self.target_angle = np.random.rand() * (2 * np.pi) - np.pi
        self.is_rotating = False

        # TP2
        self.robot_position = np.array(robot_position)
        self.goal = np.array([-400  , -50])
        self.theta_Oref = 0

        # TP4
        self.iteration = 0

        # TP5
        self.path = []
        self.path_index = 0
        self.exploration_iterations = 700
        self.path_following = False 

        # Debug window
        self.debug_window = DebugWindow()
        self.last_command = {"forward": 0.0, "rotation": 0.0}

    def transform_odom_to_world(self, odom_pose):
        """Transform a pose from odometry frame to world frame."""
        x_0, y_0, theta_0 = odom_pose
        x_Oref, y_Oref = self.robot_position
        theta_Oref = getattr(self, 'theta_Oref', 0)
        x = x_Oref + x_0 * np.cos(theta_Oref) - y_0 * np.sin(theta_Oref)
        y = y_Oref + x_0 * np.sin(theta_Oref) + y_0 * np.cos(theta_Oref)
        theta = theta_0 + theta_Oref
        return np.array([x, y, theta])

    def transform_world_to_odom(self, world_pose):
        """Transform a pose from world frame to odometry frame."""
        x, y, theta = world_pose
        x_Oref, y_Oref = self.robot_position
        theta_Oref = getattr(self, 'theta_Oref', 0)
        x_0 = (x - x_Oref) * np.cos(theta_Oref) + (y - y_Oref) * np.sin(theta_Oref)
        y_0 = -(x - x_Oref) * np.sin(theta_Oref) + (y - y_Oref) * np.cos(theta_Oref)
        theta_0 = theta - theta_Oref
        return np.array([x_0, y_0, theta_0])

    def plan_path(self, start_pose, goal_pose, corrected_pose):
        """
        Plan a path from start_pose to goal_pose using the planner and visualize it
        """
        self.path = self.planner.plan(start_pose, goal_pose)
        self.path_index = 0
        if self.path:
            traj = np.array([[p[0] for p in self.path], [p[1] for p in self.path]])
            self.tiny_slam.grid.display_cv(corrected_pose, goal=goal_pose[:2], traj=traj)
            self.debug_window.add_status_message(
                f"Path planned to origin: {len(self.path)} waypoints",
                self.debug_window.success_color
            )
            return True
        else:
            self.debug_window.add_status_message(
                "No path found to origin, stopping",
                self.debug_window.error_color
            )
            self.path_following = False
            return False

    def path_following_control(self, world_pose, corrected_pose):
        if not self.path or self.path_index >= len(self.path):
            self.debug_window.add_status_message(
                "No valid path or path completed, stopping",
                self.debug_window.error_color
            )
            return {"forward": 0, "rotation": 0}

        next_waypoint = self.path[self.path_index]
        dist_to_waypoint = np.linalg.norm(world_pose[:2] - next_waypoint[:2])
        
        if dist_to_waypoint < 20:
            self.path_index += 1
            if self.path_index < len(self.path):
                self.debug_window.add_status_message(
                    f"Reached waypoint {self.path_index}/{len(self.path)}",
                    self.debug_window.success_color
                )
            else:
                self.debug_window.add_status_message(
                    "Reached origin, stopping",
                    self.debug_window.success_color
                )
                return {"forward": 0, "rotation": 0}

        waypoint_world = np.array([next_waypoint[0], next_waypoint[1], 0])
        goal_in_odom = self.transform_world_to_odom(waypoint_world)
        command = potential_field_control(
            self.lidar(),
            corrected_pose,
            np.array([goal_in_odom[0], goal_in_odom[1], corrected_pose[2]]),
            debug_window=self.debug_window
        )
        return command

    def control(self):
        """
        Main control function executed at each time step
        """
        self.debug_window.render()
        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1 with minimal random motion
        """
        _, _, theta = self.odometer_values()
        command, self.is_rotating = reactive_obst_avoid(self.lidar(), theta, self.target_angle, self.is_rotating)
        
        # Debug rendering
        self.debug_window.update_components({
            "attractive_vel": 0.0,  # No attractive force in TP1
            "repulsive_vel": 0.0,   # No repulsive force computed explicitly
            "d_obs": np.min(self.lidar().get_sensor_values()) - 25  # Assume d_safe=25
        })
        self.debug_window.update(
            position=self.odometer_values()[:2],
            goal=np.array([0, 0]),  # No specific goal in TP1
            slam_score=0.0,
            speed=command["forward"],
            rotation=command["rotation"],
            iteration=self.iteration,
            max_range=self.lidar().max_range,
            mode="Rotating",
            attractive_vel=0.0,
            repulsive_vel=0.0
        )
        
        
        self.last_command = command
        return command

    def control_tp2(self):
        """
        Control function for TP2 with potential field
        """
        current_pose = self.odometer_values()
        world_pose = self.transform_odom_to_world(current_pose)

        goal_world = np.array([self.goal[0], self.goal[1], 0])
        goal_odom = self.transform_world_to_odom(goal_world)
        self.iteration += 1

        command = potential_field_control(self.lidar(), current_pose, goal_odom, debug_window=self.debug_window)
        
        # command = dynamic_window_control(self.lidar(), world_pose, goal_odom)
        
        self.debug_window.update(
            position=current_pose[:2],
            goal=self.goal,
            slam_score=0.0,
            speed=command["forward"],
            rotation=command["rotation"],
            iteration=self.iteration,
            max_range=self.lidar().max_range,
            mode="Potential Field",
            attractive_vel=float(self.debug_window.labels["attractive_vel"].cget("text")),
            repulsive_vel=float(self.debug_window.labels["repulsive_vel"].cget("text")),
            position_local=f"{current_pose[0]:.1f}, {current_pose[1]:.1f}, {current_pose[2]:.1f}",
            position_odom=f"{current_pose[0]:.1f}, {current_pose[1]:.1f}, {current_pose[2]:.1f}",
            position_world=f"{world_pose[0]:.1f}, {world_pose[1]:.1f}, {world_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command

    def control_tp3(self):
        """
        Control function for TP3 with SLAM
        """
        pose = self.odometer_values()
        self.tiny_slam.update_map(self.lidar(), pose, sensor_model="noisy")
        self.tiny_slam.grid.display_cv(pose)
        
        current_pose = pose
        world_pose = self.transform_odom_to_world(current_pose)

        goal_world = np.array([self.goal[0], self.goal[1], 0])
        goal_odom = self.transform_world_to_odom(goal_world)
        self.iteration += 1

        command = potential_field_control(self.lidar(), world_pose, goal_odom, debug_window=self.debug_window)
        
        self.debug_window.update(
            position=current_pose[:2],
            goal=self.goal,
            slam_score=0.0,
            speed=command["forward"],
            rotation=command["rotation"],
            iteration=self.iteration,
            max_range=self.lidar().max_range,
            mode="SLAM",
            attractive_vel=float(self.debug_window.labels["attractive_vel"].cget("text")),
            repulsive_vel=float(self.debug_window.labels["repulsive_vel"].cget("text")),
            position_local=f"{current_pose[0]:.1f}, {current_pose[1]:.1f}, {current_pose[2]:.1f}",
            position_odom=f"{current_pose[0]:.1f}, {current_pose[1]:.1f}, {current_pose[2]:.1f}",
            position_world=f"{world_pose[0]:.1f}, {world_pose[1]:.1f}, {world_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command

    def control_tp4(self):
        """
        Control function for TP4 with SLAM and localization
        """
        raw_odom = self.odometer_values()
        score = self.tiny_slam.localise(self.lidar(), raw_odom, localisation_method="cem")
        corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        if score > self.tiny_slam.score_threshold:
            old_position = self.robot_position.copy()
            self.robot_position = corrected_pose[:2]
            self.debug_window.add_status_message(
                f"Position updated: ({old_position[0]:.1f}, {old_position[1]:.1f}) → ({self.robot_position[0]:.1f}, {self.robot_position[1]:.1f})",
                self.debug_window.warning_color
            )

        self.tiny_slam.update_map(self.lidar(), corrected_pose, sensor_model="noisy")
        self.tiny_slam.grid.display_cv(corrected_pose)
        
        goal_world = np.array([self.goal[0], self.goal[1], 0])
        world_pose = self.transform_odom_to_world(corrected_pose)
        goal_odom = self.transform_world_to_odom(goal_world)

        self.iteration += 1

        command = potential_field_control(self.lidar(), world_pose, goal_odom, debug_window=self.debug_window)
        
        self.debug_window.update(
            position=world_pose[:2],
            goal=self.goal,
            slam_score=score,
            speed=command["forward"],
            rotation=command["rotation"],
            iteration=self.iteration,
            max_range=self.lidar().max_range,
            mode="SLAM+Localization",
            attractive_vel=float(self.debug_window.labels["attractive_vel"].cget("text")),
            repulsive_vel=float(self.debug_window.labels["repulsive_vel"].cget("text")),
            position_local=f"{raw_odom[0]:.1f}, {raw_odom[1]:.1f}, {raw_odom[2]:.1f}",
            position_odom=f"{corrected_pose[0]:.1f}, {corrected_pose[1]:.1f}, {corrected_pose[2]:.1f}",
            position_world=f"{world_pose[0]:.1f}, {world_pose[1]:.1f}, {world_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command

    def control_tp5(self):
        """
        Control function for TP5 with SLAM, planning, and path following
        """
        raw_odom = self.odometer_values()
        score = self.tiny_slam.localise(self.lidar(), raw_odom)
        corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        if score > self.tiny_slam.score_threshold:
            old_position = self.robot_position.copy()
            self.robot_position = corrected_pose[:2]
            self.debug_window.add_status_message(
                f"Position updated: ({old_position[0]:.1f}, {old_position[1]:.1f}) → ({self.robot_position[0]:.1f}, {self.robot_position[1]:.1f})",
                self.debug_window.warning_color
            )

        self.tiny_slam.update_map(self.lidar(), corrected_pose) # TBT
        world_pose = self.transform_odom_to_world(corrected_pose)

        self.iteration += 1

        if self.iteration <= self.exploration_iterations:
            command = self.control_tp2()
            self.path_following = False
        else:
            self.path_following = True
            if self.iteration == self.exploration_iterations + 1:
                goal = np.array([0, 0, 0])
                if not self.plan_path(world_pose, goal, corrected_pose):
                    command = {"forward": 0, "rotation": 0}
                else:
                    command = self.path_following_control(world_pose, corrected_pose)
            else:
                command = self.path_following_control(world_pose, corrected_pose)

        # Debug rendering
        goal = self.goal if not self.path_following else (self.path[self.path_index][:2] if self.path and self.path_index < len(self.path) else np.array([0, 0]))
        attractive_vel = float(self.debug_window.labels["attractive_vel"].cget("text"))
        repulsive_vel = float(self.debug_window.labels["repulsive_vel"].cget("text"))
        
        self.debug_window.update(
            position=world_pose[:2],
            goal=goal,
            slam_score=score,
            speed=command["forward"],
            rotation=command["rotation"],
            iteration=self.iteration,
            max_range=self.lidar().max_range,
            mode="Exploring" if not self.path_following else "Path Following",
            attractive_vel=attractive_vel,
            repulsive_vel=repulsive_vel,
            position_local=f"{raw_odom[0]:.1f}, {raw_odom[1]:.1f}, {raw_odom[2]:.1f}",
            position_odom=f"{corrected_pose[0]:.1f}, {corrected_pose[1]:.1f}, {corrected_pose[2]:.1f}",
            position_world=f"{world_pose[0]:.1f}, {world_pose[1]:.1f}, {world_pose[2]:.1f}"
        )
        
        # self.tiny_slam.grid.display_cv(corrected_pose)
        
        self.last_command = command
        return command