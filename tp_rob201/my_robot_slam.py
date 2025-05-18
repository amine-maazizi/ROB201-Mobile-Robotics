import numpy as np
from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams
from tiny_slam import TinySlam
from debug_window import DebugWindow
from control import potential_field_control
from occupancy_grid import OccupancyGrid
from planner import Planner
from mpi4py import MPI  # Import MPI4py for parallel processing

class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # MPI initialization
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.size != 2:
            if self.rank == 0:
                print("Error: This implementation requires exactly 2 MPI processes (1 for robot, 1 for SLAM).")
            MPI.Finalize()
            exit(1)

        # Step counter to deal with init and display
        self.counter = 0

        # Common parameters for all processes
        size_area = (1400, 1000)
        robot_position = (439.0, 195)

        # Init SLAM object (only in SLAM process, rank 1)
        if self.rank == 1:
            self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                                x_max=size_area[0] / 2 - robot_position[0],
                                                y_min=-(size_area[1] / 2 + robot_position[1]),
                                                y_max=size_area[1] / 2 - robot_position[1],
                                                resolution=2)
            self.tiny_slam = TinySlam(self.occupancy_grid)
            # Start SLAM process
            self.run_slam_process()
            return  # Exit initialization for SLAM process
        else:
            self.occupancy_grid = None
            self.tiny_slam = None

        # Common attributes
        if self.rank == 0:
            self.planner = Planner(None)  # Planner will use map data from SLAM process if needed
            self.debug_window = DebugWindow()
            self.last_command = {"forward": 0.0, "rotation": 0.0}
        else:
            self.planner = None
            self.debug_window = None
            self.last_command = None

        # Storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # TP1
        self.target_angle = np.random.rand() * (2 * np.pi) - np.pi
        self.is_rotating = False

        # TP2
        self.robot_pose_odom = np.array([0, 0, 0])
        self.initial_robot_position = np.array(robot_position)
        self.goal_odom = np.array([-500, -50, 0])

        # TP4
        self.iteration = 0

        # TP5
        self.path = []
        self.path_index = 0
        self.exploration_iterations = 750
        self.path_following = False

    def run_slam_process(self):
        """SLAM process (rank 1) that handles localization, map updating, and rendering"""
        while True:
            # Receive LIDAR and odometry data from robot process
            data = self.comm.recv(source=0, tag=0)
            if data is None:  # Termination signal
                break

            lidar_data, lidar_angles, max_range, raw_odom = data
            
            # Create a simple object with the required attributes
            class LidarData:
                def __init__(self, values, angles, max_range):
                    self.values = values
                    self.angles = angles
                    self.max_range = max_range
                
                def get_sensor_values(self):
                    return self.values
                
                def get_ray_angles(self):
                    return self.angles

            lidar = LidarData(lidar_data, lidar_angles, max_range)
            
            # Perform localization
            score = self.tiny_slam.localise(lidar, raw_odom)
            corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

            # Update map
            self.tiny_slam.update_map(lidar, corrected_pose)

            # Send back score and corrected pose
            self.comm.send((score, corrected_pose), dest=0, tag=1)

        # Finalize SLAM process
        MPI.Finalize()

    def control(self):
        """Main control function executed at each time step"""
        if self.rank == 1:
            return {"forward": 0, "rotation": 0}  # SLAM process doesn't control robot
        else:
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
        """Plan a path from start_pose to goal_pose. Map data is managed by SLAM process."""
        # In a real implementation, the planner would need map data from the SLAM process.
        # For simplicity, assume planner uses a precomputed map or communicates internally.
        self.path = self.planner.plan(start_pose_odom, goal_pose_odom)
        self.path_index = 0
        if self.path:
            traj = np.array([[p[0] for p in self.path], [p[1] for p in self.path]])
            # Note: Map display is handled by SLAM process
            path_str = " -> ".join([f"({p[0]:.1f}, {p[1]:.1f})" for p in self.path])
            self.debug_window.add_status_message(
                f"Path planned: {path_str}",
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
        """Follow the path using a simple local controller."""
        if not self.path or self.path_index >= len(self.path):
            self.debug_window.add_status_message(
                "No valid path or path completed, stopping",
                self.debug_window.error_color
            )
            return {"forward": 0, "rotation": 0}

        next_waypoint = self.path[self.path_index]
        dist_to_waypoint = np.linalg.norm(corrected_pose[:2] - next_waypoint[:2])
        
        self.debug_window.add_status_message(
            f"Current waypoint {self.path_index}/{len(self.path)}: ({next_waypoint[0]:.1f}, {next_waypoint[1]:.1f}), Distance: {dist_to_waypoint:.1f}",
            self.debug_window.info_color
        )
        
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

        dx = next_waypoint[0] - corrected_pose[0]
        dy = next_waypoint[1] - corrected_pose[1]
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - corrected_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        K_angle = 1.0
        K_dist = 0.5
        rotation = K_angle * angle_diff
        forward = K_dist * dist_to_waypoint
        
        if abs(angle_diff) > np.pi/4:
            forward *= 0.5
        
        max_forward = 0.5
        max_rotation = 1.0
        forward = np.clip(forward, -max_forward, max_forward)
        rotation = np.clip(rotation, -max_rotation, max_rotation)
        
        return {"forward": forward, "rotation": rotation}

    def control_tp5(self):
        """Control function for TP5 with SLAM in a separate MPI process."""
        raw_odom = self.odometer_values()
        lidar = self.lidar()
        
        # Extract the data we need to send
        lidar_data = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()
        max_range = lidar.max_range

        # Send LIDAR and odometry data to SLAM process (rank 1)
        self.comm.send((lidar_data, lidar_angles, max_range, raw_odom), dest=1, tag=0)

        # Receive SLAM results
        score, corrected_pose = self.comm.recv(source=1, tag=1)

        # Update pose if score is above threshold
        if score > self.tiny_slam.score_threshold:
            self.robot_pose_odom = corrected_pose.copy()
            self.debug_window.add_status_message(
                f"Pose updated: ({raw_odom[0]:.1f}, {raw_odom[1]:.1f}) → ({corrected_pose[0]:.1f}, {corrected_pose[1]:.1f})",
                self.debug_window.warning_color
            )

        self.iteration += 1

        # Control logic (exploration or path following)
        if self.iteration <= self.exploration_iterations:
            command = potential_field_control(lidar, corrected_pose, self.goal_odom, debug_window=self.debug_window)
            self.path_following = False
        else:
            self.path_following = True
            if self.iteration == self.exploration_iterations + 1:
                goal_pose_odom = np.array([0, 0, 0])
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
        
        if self.path_following and self.path and self.path_index < len(self.path):
            dist_to_waypoint = np.linalg.norm(corrected_pose[:2] - self.path[self.path_index][:2])
            self.debug_window.update_components({
                "distance": dist_to_waypoint
            })
        
        self.debug_window.update(
            position=corrected_pose[:2],
            goal=goal,
            slam_score=score,
            speed=command["forward"],
            rotation=command["rotation"],
            iteration=self.iteration,
            max_range=max_range,
            mode="Exploring" if not self.path_following else "Path Following",
            attractive_vel=attractive_vel,
            repulsive_vel=repulsive_vel,
            position_local=f"{raw_odom[0]:.1f}, {raw_odom[1]:.1f}, {raw_odom[2]:.1f}",
            position_odom=f"{corrected_pose[0]:.1f}, {corrected_pose[1]:.1f}, {corrected_pose[2]:.1f}",
            position_world=f"{self.initial_robot_position[0]+corrected_pose[0]:.1f}, {self.initial_robot_position[1]+corrected_pose[1]:.1f}, {corrected_pose[2]:.1f}"
        )
        
        self.last_command = command
        return command