import sys
import numpy as np
from car import Car
from particle_filter import ParticleFilter
from kalman_filter import KalmanFilter
from utils import angle_bw
from racetrack import load_racetrack

# EXTRA CREDIT: Import extended filters
try:
    from particle_filter_extra import ParticleFilterExtra
    EXTRA_CREDIT_AVAILABLE = True
except ImportError:
    EXTRA_CREDIT_AVAILABLE = False
    print("Note: Extra credit modules not fully implemented yet")

WORLD_WIDTH = 1400
WORLD_HEIGHT = 800
CAR_LENGTH = 40
CAR_WIDTH = 20

class Simulator:
    def __init__(self, max_sensor_range=50, sensor_std=0.0, num_particles=50, gps_noise_var=10.0, gps_noise_width=20, noise_type="gaussian"):
        self.racetrack = load_racetrack("data/racetrack.p")
        self.car1 = Car(max_sensor_range=max_sensor_range, sensor_std=sensor_std, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width)
        self.car2 = Car(max_sensor_range=max_sensor_range, sensor_std=sensor_std, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width)
        
        # Set different initial positions for car1 and car2
        self.car1.pos = np.array([750.0, 760.0])
        self.car2.pos = np.array([750.0, 730.0])
        
        self.max_sensor_range = max_sensor_range
        self.sensor_std = sensor_std
        self.num_particles = num_particles
        self.do_particle_filtering = False
        self.particle_filter1 = None
        self.particle_filter2 = None
        self.x_est1 = self.y_est1 = self.orient_est1 = None
        self.x_est2 = self.y_est2 = self.orient_est2 = None

        self.do_kalman_filtering = False
        self.kalman_filter1 = None
        self.kalman_filter2 = None
        self.kf_state1 = None
        self.kf_state2 = None
        self.gps_noise_var = gps_noise_var
        self.gps_noise_width = gps_noise_width
        self.gps_noise_dist = noise_type
        
        # EXTRA CREDIT: Support for different noise distributions
        # Automatically use extended filters for laplace/cauchy noise
        self.pf_noise_type = noise_type  # For particle filter transition noise
        self.sensor_noise_type = noise_type  # For sensor measurement noise

        self.r_count = 0
        self.cur_rightness = 0.5

        self.lap_data = []
        self.crossed_start = False
        self.lap_data_old = np.load("data/lap_data.npy")
        self.cur_i = 2
        self.recording = False
        self.replaying = False

        # Initialize checkpoints and related data
        self.checkpoints = [
            (1000, 750), (1250, 700), (1350, 300), (1250, 100), 
            (900, 250), (500, 300), (100, 100), (300, 700), (750, 750)
        ]
        self.car1_next_checkpoint = 0
        self.car2_next_checkpoint = 0
        self.car1_checkpoint_errors = []
        self.car2_checkpoint_errors = []
        self.car1_checkpoint_reached = []
        self.car2_checkpoint_reached = []
        self.car1_laps = 0
        self.car2_laps = 0

        self.game_over = False
        self.winner = None

    def get_next_checkpoint(self, car_num):
        """Get the next checkpoint coordinates for the specified car"""
        if car_num == 1:
            return self.checkpoints[self.car1_next_checkpoint]
        else:
            return self.checkpoints[self.car2_next_checkpoint]

    def calculate_checkpoint_error(self, car_pos, est_pos, checkpoint_pos):
        """Calculate error between estimated position and actual position relative to checkpoint"""
        for k in est_pos:
            if k is None:  # Add check for None
                return 0.0
        car_to_checkpoint = np.array(checkpoint_pos) - np.array(car_pos)
        est_to_checkpoint = np.array(checkpoint_pos) - np.array(est_pos)
        return np.linalg.norm(car_to_checkpoint - est_to_checkpoint)

    def check_collision(self):
        # Get the corners of both cars to define their bounding boxes
        car1_corners = self.get_car_corners(self.car1)
        car2_corners = self.get_car_corners(self.car2)

        # Check if the bounding boxes of the two cars overlap
        if self.rectangles_collide(car1_corners, car2_corners):
            # Calculate collision normal vector (direction from car1 to car2)
            collision_normal = self.car2.pos - self.car1.pos
            
            # Normalize the collision vector
            norm = np.linalg.norm(collision_normal)
            if norm > 1e-10:
                collision_normal /= norm
            else:
                # Fallback to avoid division by zero
                collision_normal = np.array([1.0, 0.0])

            # Calculate relative velocity between the two cars
            relative_velocity = self.car2.vel - self.car1.vel
            
            # Calculate impulse (change in momentum) assuming elastic collision with equal mass
            impulse = 2 * np.dot(relative_velocity, collision_normal) * collision_normal
            
            # Apply impulse to change velocities (Newton's 3rd law)
            self.car1.vel += impulse / 2
            self.car2.vel -= impulse / 2

            # Apply damping/friction to simulate inelastic collision (loss of energy)
            self.car1.vel *= 0.8
            self.car2.vel *= 0.8
            
            # Update old velocities to match current velocities
            self.car1.old_vel = self.car1.vel
            self.car2.old_vel = self.car2.vel

    def get_car_corners(self, car):
        half_length = CAR_LENGTH / 2
        half_width = CAR_WIDTH / 2
        cos_theta = car.orient[0]
        sin_theta = car.orient[1]

        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        rotated_corners = np.dot(corners, rotation_matrix.T)
        return rotated_corners + car.pos

    def rectangles_collide(self, corners1, corners2):
        for shape in [corners1, corners2]:
            for i in range(len(shape)):
                axis = np.array([-shape[i-1][1] + shape[i][1], shape[i-1][0] - shape[i][0]])
                axis /= np.linalg.norm(axis)

                min1, max1 = float('inf'), float('-inf')
                min2, max2 = float('inf'), float('-inf')

                for corner in corners1:
                    projection = np.dot(corner, axis)
                    min1 = min(min1, projection)
                    max1 = max(max1, projection)

                for corner in corners2:
                    projection = np.dot(corner, axis)
                    min2 = min(min2, projection)
                    max2 = max(max2, projection)

                if max1 < min2 or max2 < min1:
                    return False

        return True

    def update_race_progress(self):
        # Iterate over both cars to check their progress
        for i, (car, next_checkpoint) in enumerate([(self.car1, self.car1_next_checkpoint), 
                                                  (self.car2, self.car2_next_checkpoint)]):
            checkpoint = self.checkpoints[next_checkpoint]
            
            # Check if the car is within range (30 units) of its target checkpoint
            if np.linalg.norm(car.pos - checkpoint) < 30:
                if i == 0:
                    # Update Car 1's progress
                    self.car1_checkpoint_reached.append((next_checkpoint, self.car1.pos.copy()))
                    self.car1_next_checkpoint = (self.car1_next_checkpoint + 1) % len(self.checkpoints)
                    
                    # Check for lap completion
                    if self.car1_next_checkpoint == 0:
                        self.car1_laps += 1
                        # Check for win condition (3 laps)
                        if self.car1_laps == 3:
                            self.game_over = True
                            self.winner = "Car 1"
                else:
                    # Update Car 2's progress
                    self.car2_checkpoint_reached.append((next_checkpoint, self.car2.pos.copy()))
                    self.car2_next_checkpoint = (self.car2_next_checkpoint + 1) % len(self.checkpoints)
                    
                    # Check for lap completion
                    if self.car2_next_checkpoint == 0:
                        self.car2_laps += 1
                        # Check for win condition (3 laps)
                        if self.car2_laps == 3:
                            self.game_over = True
                            self.winner = "Car 2"

        # Calculate checkpoint errors
        if self.do_particle_filtering:
            checkpoint = self.get_next_checkpoint(1)
            error1 = self.calculate_checkpoint_error(
                self.car1.pos, 
                np.array([self.x_est1, self.y_est1]), 
                checkpoint
            )
            self.car1_checkpoint_errors.append(error1)


            error2 = self.calculate_checkpoint_error(
                self.car1.pos, 
                np.array([self.x_est2, self.y_est2]), 
                checkpoint
            )
            self.car2_checkpoint_errors.append(error2)
            
        if self.do_kalman_filtering:
            checkpoint1 = self.get_next_checkpoint(1)
            error1 = self.calculate_checkpoint_error(
                self.car1.pos,
                self.kf_state1[:2],
                checkpoint1
            )
            self.car1_checkpoint_errors.append(error1)
            
            checkpoint2 = self.get_next_checkpoint(2)
            error2 = self.calculate_checkpoint_error(
                self.car2.pos,
                self.kf_state2[:2],
                checkpoint2
            )
            self.car2_checkpoint_errors.append(error2)

    def init_particles(self):
        self.do_particle_filtering = True
        # EXTRA CREDIT: Use extended particle filter for laplace/cauchy noise
        if self.pf_noise_type in ["laplace", "cauchy"] and EXTRA_CREDIT_AVAILABLE:
            self.particle_filter1 = ParticleFilterExtra(self.num_particles, 0, WORLD_WIDTH, 0, WORLD_HEIGHT, noise_type=self.pf_noise_type)
            self.particle_filter2 = ParticleFilterExtra(self.num_particles, 0, WORLD_WIDTH, 0, WORLD_HEIGHT, noise_type=self.pf_noise_type)
        else:
            self.particle_filter1 = ParticleFilter(self.num_particles, 0, WORLD_WIDTH, 0, WORLD_HEIGHT)
            self.particle_filter2 = ParticleFilter(self.num_particles, 0, WORLD_WIDTH, 0, WORLD_HEIGHT)
    
    def stop_particles(self):
        self.do_particle_filtering = False
        self.particle_filter1 = None
        self.particle_filter2 = None
    
    def init_kalman(self):
        self.do_kalman_filtering = True
        
        # Adjust variance for different noise distributions
        if self.gps_noise_dist == "laplace":
            # For Laplace: Use larger variance to show heavier tails clearly
            adjusted_var = 10 * self.car1.gps_noise_var
        elif self.gps_noise_dist == "cauchy":
            # For Cauchy: variance undefined, use larger value for heavy tails
            adjusted_var = 10 * self.car1.gps_noise_var
        else:
            # Gaussian or uniform: use variance as-is
            adjusted_var = self.car1.gps_noise_var
        
        # Use base KalmanFilter with adjusted variance (works for all noise types)
        self.kalman_filter1 = KalmanFilter(self.car1, adjusted_var, self.gps_noise_width)
        self.kalman_filter2 = KalmanFilter(self.car2, adjusted_var, self.gps_noise_width)
    
    def stop_kalman(self):
        self.do_kalman_filtering = False
        self.kalman_filter1 = self.kalman_filter2 = None
    
    def toggle_particles(self):
        if self.do_particle_filtering:
            self.stop_particles()
        else:
            self.init_particles()
    
    def toggle_kalman(self):
        if self.do_kalman_filtering:
            self.stop_kalman()
        else:
            self.init_kalman()
        
    def toggle_gps_noise_dist(self):
        # EXTRA CREDIT: Cycle through all noise distributions if available
        if EXTRA_CREDIT_AVAILABLE:
            noise_types = ["gaussian", "uniform", "laplace", "cauchy"]
        else:
            noise_types = ["gaussian", "uniform"]
        
        current_idx = noise_types.index(self.gps_noise_dist) if self.gps_noise_dist in noise_types else 0
        self.gps_noise_dist = noise_types[(current_idx + 1) % len(noise_types)]
        
        # Update particle filter noise type too
        self.pf_noise_type = self.gps_noise_dist if self.gps_noise_dist != "uniform" else "gaussian"
        
        # Reinitialize filters with new noise type
        if self.do_particle_filtering:
            self.stop_particles()
            self.init_particles()
        if self.do_kalman_filtering:
            self.stop_kalman()
            self.init_kalman()
    
    def toggle_replay(self):
        self.replaying = not self.replaying
    
    def loop(self):
        # Recording mode: save car positions and orientations
        if self.recording:
            self.lap_data.append(np.append(self.car1.pos, self.car1.orient))
            self.lap_data.append(np.append(self.car2.pos, self.car2.orient))
            
            datum1 = self.racetrack.progress(self.car1)
            datum2 = self.racetrack.progress(self.car2)
            progress = max(datum1[0], datum2[0])
            # Check if crossed start/finish line to complete a lap recording
            if 1.1 < progress < 1.5 and not self.crossed_start:
                self.crossed_start = True
            if 0.5 < progress < 1.0 and self.crossed_start:
                np.save("data/lap_data.npy", np.array(self.lap_data))
                print("finished")
                self.crossed_start = False

        # Replay mode: playback recorded data
        if self.replaying:
            if self.cur_i >= len(self.lap_data_old) - 1:
                print("replay finished")
                return False
            
            # Update car state from recorded data
            dp1 = self.lap_data_old[self.cur_i]
            dp2 = self.lap_data_old[self.cur_i + 1]
            
            self.car1.pos = dp1[0:2]
            self.car1.orient = dp1[2:]
            self.car2.pos = dp2[0:2]
            self.car2.orient = dp2[2:]

            # Reconstruct velocity from position changes
            if self.cur_i >= 2:
                dp1_prev = self.lap_data_old[self.cur_i - 2]
                dp2_prev = self.lap_data_old[self.cur_i - 1]
                self.car1.vel = self.car1.pos - dp1_prev[0:2]
                self.car1.old_vel = dp1_prev[0:2] - self.lap_data_old[self.cur_i - 4][0:2]
                self.car2.vel = self.car2.pos - dp2_prev[0:2]
                self.car2.old_vel = dp2_prev[0:2] - self.lap_data_old[self.cur_i - 3][0:2]

            self.cur_i += 2
        else:
            # Standard simulation update: update physics and controls
            self.car1.update(self.racetrack.contour_inner, self.racetrack.contour_outer)
            self.car2.update(self.racetrack.contour_inner, self.racetrack.contour_outer)

        # Calculate change in orientation
        d_orient1 = angle_bw(self.car1.vel, self.car1.old_vel) * np.pi / 180.0
        d_orient2 = angle_bw(self.car2.vel, self.car2.old_vel) * np.pi / 180.0

        # Simulate sensor readings
        self.car1.measure_sensor_dists(self.racetrack)
        self.car2.measure_sensor_dists(self.racetrack)

        # Run Particle Filter update if enabled
        if self.do_particle_filtering:
            sensor = self.racetrack.read_distances
            
            evidence1 = self.car1.sensor_dists
            evidence2 = self.car2.sensor_dists
            speed1 = np.linalg.norm(self.car1.vel)
            speed2 = np.linalg.norm(self.car2.vel)
            self.x_est1, self.y_est1, self.orient_est1 = self.particle_filter1.filtering_and_estimation(
                sensor, self.max_sensor_range, self.sensor_std, evidence1, d_orient1, speed1)
            self.x_est2, self.y_est2, self.orient_est2 = self.particle_filter2.filtering_and_estimation(
                sensor, self.max_sensor_range, self.sensor_std, evidence2, d_orient2, speed2)

        # Run Kalman Filter update if enabled
        if self.do_kalman_filtering:
            self.gps_measurement1 = self.car1.measure_gps(noise_dist=self.gps_noise_dist)
            self.kf_state1 = self.kalman_filter1.predict_and_update(
                self.gps_measurement1, self.car2, self.gps_noise_dist)

            self.gps_measurement2 = self.car2.measure_gps(noise_dist=self.gps_noise_dist)
            self.kf_state2 = self.kalman_filter2.predict_and_update(
                self.gps_measurement2, self.car1, self.gps_noise_dist)
            
        # Check for collisions and update race status
        self.check_collision()
        self.update_race_progress()

        if self.game_over:
            print(f"Game Over! {self.winner} wins!")
            return False

        return True
