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
        self.robot_pose_odom = np.array([0, 0, 0])  # Initialize with full pose in odom frame
        self.initial_robot_position = np.array(robot_position)  # World initial position for display
        self.goal_odom = np.array([-400, -50, 0])  # Goal in odom frame
        
        # TP4
        self.iteration = 0

        # TP5
        self.path = []
        self.path_index = 0
        self.exploration_iterations = 1000
        self.path_following = False 

        # Debug window
        self.debug_window = DebugWindow()
        self.last_command = {"forward": 0.0, "rotation": 0.0}

    def control(self):
        """
        Main control function executed at each time step
        """
        self.debug_window.render()
        return self.control_tp5()

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
        
        # Everything in odom frame
        self.iteration += 1
        command = potential_field_control(self.lidar(), current_pose, self.goal_odom, debug_window=self.debug_window)
        
        self.debug_window.update(
            position=current_pose[:2],
            goal=self.goal_odom[:2],
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
            position_world=f"{self.initial_robot_position[0]+current_pose[0]:.1f}, {self.initial_robot_position[1]+current_pose[1]:.1f}, {current_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command

    def control_tp3(self):
        """
        Control function for TP3 with SLAM
        """
        current_pose = self.odometer_values()
        
        # Use current_pose (odom frame) for SLAM - keeping everything in odom frame
        self.tiny_slam.update_map(self.lidar(), current_pose, sensor_model="noisy")
        self.tiny_slam.grid.display_cv(current_pose)
        
        self.iteration += 1
        command = potential_field_control(self.lidar(), current_pose, self.goal_odom, debug_window=self.debug_window)
        
        self.debug_window.update(
            position=current_pose[:2],
            goal=self.goal_odom[:2],
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
            position_world=f"{self.initial_robot_position[0]+current_pose[0]:.1f}, {self.initial_robot_position[1]+current_pose[1]:.1f}, {current_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command

    def control_tp4(self):
        """
        Control function for TP4 with SLAM and localization
        """
        raw_odom = self.odometer_values()
        score = self.tiny_slam.localise(self.lidar(), raw_odom, localisation_method="cma-es")
        corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)
        
        if score > self.tiny_slam.score_threshold:
            self.robot_pose_odom = corrected_pose.copy()  # Update full pose in odom frame
            self.debug_window.add_status_message(
                f"Pose updated: ({raw_odom[0]:.1f}, {raw_odom[1]:.1f}) → ({corrected_pose[0]:.1f}, {corrected_pose[1]:.1f})",
                self.debug_window.warning_color
            )

        # Update map with corrected pose (odom frame)
        self.tiny_slam.update_map(self.lidar(), corrected_pose, sensor_model="noisy")
        self.tiny_slam.grid.display_cv(corrected_pose)
        
        self.iteration += 1

        # Use corrected_pose and goal_odom - both in odom frame
        command = potential_field_control(self.lidar(), corrected_pose, self.goal_odom, debug_window=self.debug_window)
        
        self.debug_window.update(
            position=corrected_pose[:2],
            goal=self.goal_odom[:2],
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
            position_world=f"{self.initial_robot_position[0]+corrected_pose[0]:.1f}, {self.initial_robot_position[1]+corrected_pose[1]:.1f}, {corrected_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command

    def plan_path(self, start_pose_odom, goal_pose_odom, corrected_pose):
        """
        Plan a path from start_pose to goal_pose using the planner and visualize it.
        All poses are in odom frame.
        """
        self.path = self.planner.plan(start_pose_odom, goal_pose_odom)
        self.path_index = 0
        if self.path:
            traj = np.array([[p[0] for p in self.path], [p[1] for p in self.path]])
            self.tiny_slam.grid.display_cv(corrected_pose, goal=goal_pose_odom[:2], traj=traj)
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

    def path_following_control(self, corrected_pose):
        """
        Follow the path using potential field control.
        All poses and waypoints are in odom frame.
        """
        if not self.path or self.path_index >= len(self.path):
            self.debug_window.add_status_message(
                "No valid path or path completed, stopping",
                self.debug_window.error_color
            )
            return {"forward": 0, "rotation": 0}

        next_waypoint = self.path[self.path_index]
        dist_to_waypoint = np.linalg.norm(corrected_pose[:2] - next_waypoint[:2])
        
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

        # Create waypoint pose with correct orientation for potential field control
        waypoint_pose_odom = np.array([next_waypoint[0], next_waypoint[1], corrected_pose[2]])
        
        # Both corrected_pose and waypoint_pose_odom are in odom frame
        command = potential_field_control(
            self.lidar(),
            corrected_pose,
            waypoint_pose_odom,
            debug_window=self.debug_window
        )
        return command

    def control_tp5(self):
        """
        Control function for TP5 with SLAM, planning, and path following.
        All poses and waypoints are in odom frame.
        """
        raw_odom = self.odometer_values()
        score = self.tiny_slam.localise(self.lidar(), raw_odom)
        corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        if score > self.tiny_slam.score_threshold:
            self.robot_pose_odom = corrected_pose.copy()  # Store full pose
            self.debug_window.add_status_message(
                f"Pose updated: ({raw_odom[0]:.1f}, {raw_odom[1]:.1f}) → ({corrected_pose[0]:.1f}, {corrected_pose[1]:.1f})",
                self.debug_window.warning_color
            )

        # Update map with corrected pose in odom frame
        self.tiny_slam.update_map(self.lidar(), corrected_pose)
        self.iteration += 1

        if self.iteration <= self.exploration_iterations:
            # During exploration phase
            command = potential_field_control(self.lidar(), corrected_pose, self.goal_odom, debug_window=self.debug_window)
            # command = dynamic_window_control(self.lidar(), corrected_pose, self.goal_odom,
            #                                  self.last_command["forward"],
            #                                  self.last_command["rotation"])
            self.path_following = False
        else:
            # Path following phase
            self.path_following = True
            if self.iteration == self.exploration_iterations + 1:
                # Plan path to origin when we first enter path following mode
                goal_pose_odom = np.array([0, 0, 0])  # Origin in odom frame
                if not self.plan_path(corrected_pose, goal_pose_odom, corrected_pose):
                    command = {"forward": 0, "rotation": 0}
                else:
                    command = self.path_following_control(corrected_pose)
            else:
                command = self.path_following_control(corrected_pose)

        # Debug rendering
        goal = self.goal_odom[:2] if not self.path_following else (
            self.path[self.path_index][:2] if self.path and self.path_index < len(self.path) else np.array([0, 0])
        )
        
        attractive_vel = float(self.debug_window.labels["attractive_vel"].cget("text"))
        repulsive_vel = float(self.debug_window.labels["repulsive_vel"].cget("text"))
        
        self.debug_window.update(
            position=corrected_pose[:2],
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
            position_world=f"{self.initial_robot_position[0]+corrected_pose[0]:.1f}, {self.initial_robot_position[1]+corrected_pose[1]:.1f}, {corrected_pose[2]:.1f}"
        )
        
        # Display updated map
        # self.tiny_slam.grid.display_cv(corrected_pose)
        
        self.last_command = command
        return command