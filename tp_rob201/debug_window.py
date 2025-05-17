import tkinter as tk
from tkinter import ttk
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from datetime import datetime, timedelta

class DebugWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Debug Window")
        
        # Configure dark theme colors
        self.bg_color = "#2E3440"  # Dark background
        self.fg_color = "#ECEFF4"  # Light text
        self.accent_color = "#88C0D0"  # Accent color
        self.success_color = "#A3BE8C"  # Green
        self.warning_color = "#EBCB8B"  # Yellow
        self.error_color = "#BF616A"   # Red
        self.info_color = "#81A1C1"    # Blue
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        self.root.option_add('*TButton*padding', 5)
        self.root.option_add('*TLabel*padding', 5)
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("Custom.TLabel", 
                           background=self.bg_color,
                           foreground=self.fg_color,
                           font=('Consolas', 10, 'bold'))
        self.style.configure("Title.TLabel",
                           background=self.bg_color,
                           foreground=self.accent_color,
                           font=('Consolas', 10, 'bold'))
        self.style.configure("Status.TLabel",
                           background=self.bg_color,
                           foreground=self.info_color,
                           font=('Consolas', 9))
        
        # Create labels for each piece of information
        self.labels = {}
        
        # Robot Positions in different frames
        self.create_label("Robot Pos (Local)", "position_local", "0, 0, 0")
        self.create_label("Robot Pos (Odom)", "position_odom", "0, 0, 0")
        self.create_label("Robot Pos (World)", "position_world", "0, 0, 0")
        
    
        # Goal Position
        self.create_label("Goal Position", "goal", "0, 0")
        
        # Distance to Goal
        self.create_label("Distance to Goal", "distance", "0.0")
        
        # Current Speed
        self.create_label("Current Speed", "speed", "0.0")
        
        # Current Rotation
        self.create_label("Current Rotation", "rotation", "0.0")

        # Attractive Velocity (after rotation)
        self.create_label("Attractive Velocity", "attractive_vel", "0.0")
        # Repulsive Velocity (after attractive)
        self.create_label("Repulsive Velocity", "repulsive_vel", "0.0")
        # d_obs (after repulsive)
        self.create_label("d_obs", "d_obs", "0.0")

        # SLAM Score
        self.create_label("SLAM Score", "slam_score", "0")
        
        # Iteration Count
        self.create_label("Iteration Count", "iteration", "0")
        
        # Robot Mode
        self.create_label("Robot Mode", "mode", "Exploring")

        # Status bar
        self.status_frame = ttk.Frame(self.main_frame, style="Custom.TFrame")
        self.status_frame.grid(row=len(self.labels)+1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(self.status_frame, 
                                    text="Ready", 
                                    style="Status.TLabel",
                                    wraplength=400)
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Message queue for status updates
        self.message_queue = []
        self.current_message = None
        self.message_start_time = None
        self.message_duration = timedelta(seconds=3)  # Messages stay for 3 seconds
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize state
        self.position = np.array([0.0, 0.0])
        self.goal = np.array([0.0, 0.0])
        self.speed = 0.0
        self.rotation = 0.0
        self.slam_score = 0
        self.iteration = 0
        self.score_threshold = 4000
        self.max_range = 1000  # Default value, will be updated
        self.mode = "Exploring"  # Default mode

    def create_label(self, title, key, initial_value):
        """Helper method to create a label with title and value"""
        frame = ttk.Frame(self.main_frame, style="Custom.TFrame")
        frame.grid(sticky=(tk.W, tk.E), pady=2)
        
        title_label = ttk.Label(frame, text=f"{title}:", 
                              style="Title.TLabel", width=20)
        title_label.grid(row=0, column=0, padx=5)
        
        value_label = ttk.Label(frame, text=initial_value, 
                              style="Custom.TLabel", width=20)
        value_label.grid(row=0, column=1, padx=5)
        
        self.labels[key] = value_label
        
    def get_color_gradient(self, value, min_val, max_val):
        """Convert a value to a color in the gradient from green to red"""
        # Normalize value between 0 and 1
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))  # Clamp between 0 and 1
        
        # Convert to HSV (green is 120 degrees, red is 0 degrees)
        h = (1 - normalized) * 120  # Invert so green is good (0) and red is bad (1)
        s = 0.8  # High saturation
        v = 0.9  # High value
        
        # Convert HSV to RGB
        rgb = hsv_to_rgb(h/360, s, v)
        # Convert to hex color
        return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
    
    def add_status_message(self, message, color=None):
        """Add a message to the status queue"""
        if color is None:
            color = self.info_color
        self.message_queue.append((message, color))
        
    def update_status(self):
        """Update the status message display"""
        current_time = datetime.now()
        
        # If no current message or current message has expired
        if (self.current_message is None or 
            self.message_start_time is None or 
            current_time - self.message_start_time > self.message_duration):
            
            # Get next message from queue
            if self.message_queue:
                self.current_message, color = self.message_queue.pop(0)
                self.message_start_time = current_time
                self.status_label.config(text=self.current_message, foreground=color)
            else:
                self.current_message = None
                self.message_start_time = None
                self.status_label.config(text="Ready", foreground=self.info_color)
        
    def update(self, position, goal, speed, rotation, slam_score, iteration, max_range=None, mode=None, attractive_vel=None, repulsive_vel=None, position_local=None, position_odom=None, position_world=None):
        """Update the values displayed in the window with explicit parameters"""
        try:
            # Store state
            self.position = position
            self.goal = goal
            self.speed = speed
            self.rotation = rotation
            self.slam_score = slam_score
            self.iteration = iteration
            if max_range is not None:
                self.max_range = max_range
            if mode is not None:
                self.mode = mode
            if attractive_vel is not None:
                self.labels["attractive_vel"].config(text=f"{attractive_vel:.3f}")
            if repulsive_vel is not None:
                self.labels["repulsive_vel"].config(text=f"{repulsive_vel:.3f}")
            
            # Update position frames if provided
            if position_local is not None:
                self.labels["position_local"].config(text=position_local)
            if position_odom is not None:
                self.labels["position_odom"].config(text=position_odom)
            if position_world is not None:
                self.labels["position_world"].config(text=position_world)
            
            # Update goal
            self.labels["goal"].config(
                text=f"{goal[0]:.2f}, {goal[1]:.2f}"
            )
            
            # Update distance to goal with color gradient
            dist = np.linalg.norm(goal - position)
            dist_color = self.get_color_gradient(dist, 0, self.max_range)
            self.labels["distance"].config(
                text=f"{dist:.2f}",
                foreground=dist_color
            )
            
            # Update speed and rotation
            self.labels["speed"].config(text=f"{speed:.2f}")
            self.labels["rotation"].config(text=f"{rotation:.2f}")
            
            # Update SLAM score with color based on threshold
            score_color = self.success_color if slam_score > self.score_threshold else self.error_color
            self.labels["slam_score"].config(
                text=f"{slam_score:.0f}",
                foreground=score_color
            )
            
            # Update iteration count
            self.labels["iteration"].config(text=str(iteration))
            
            # Update robot mode
            mode_color = self.success_color if self.mode == "Path Following" else self.info_color
            self.labels["mode"].config(
                text=self.mode,
                foreground=mode_color
            )
            
            # Update status messages
            self.update_status()
            
        except Exception as e:
            print(f"Error updating debug window: {e}")
        
    def render(self):
        """Update the window display"""
        try:
            self.root.update()
        except Exception as e:
            print(f"Error rendering debug window: {e}")
        
    def close(self):
        """Close the debug window"""
        try:
            self.root.destroy()
        except Exception as e:
            print(f"Error closing debug window: {e}")

    def update_component(self, name, value):
        """Update the value of a label by name (if it exists)."""
        if name in self.labels:
            if isinstance(value, (int, float)):
                self.labels[name].config(text=f"{value:.3f}")
            else:
                self.labels[name].config(text=str(value))

    def update_components(self, components: dict):
        """Update multiple label values by name using a dictionary."""
        for name, value in components.items():
            if isinstance(value, (int, float)):
                self.labels[name].config(text=f"{value:.3f}")
            else:
                self.labels[name].config(text=str(value))