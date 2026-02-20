import argparse
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from simulator import Simulator, WORLD_WIDTH, WORLD_HEIGHT
from racetrack import RaceTrack, Contour, Horizontals, load_racetrack

LAP_TRIM_IDX = 690

def convert(x,y):
    new_x = np.array(x[:LAP_TRIM_IDX])
    new_y = 800 - np.array(y[:LAP_TRIM_IDX])
    return new_x, new_y

def plot_checkpoints(ax, checkpoints, reached_checkpoints=None):
    """Plot checkpoints and mark reached ones"""
    checkpoints = np.array(checkpoints)
    ax.scatter(checkpoints[:, 0], 800 - checkpoints[:, 1], 
              color='green', marker='*', s=100, label='Checkpoints')
    
    if reached_checkpoints:
        reached = np.array([pos for _, pos in reached_checkpoints])
        if len(reached) > 0:  # Only plot if there are reached checkpoints
            ax.scatter(reached[:, 0], 800 - reached[:, 1], 
                      color='yellow', marker='o', s=50, label='Reached')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--which", help="which filter to run (write pf for particle filter, kf for Kalman filter)")
    parser.add_argument("-n", "--num_particles", default=50, type=int, help='Number of particles for particle filtering')
    parser.add_argument("-m", "--max_sensor_range", default=50, type=int, help='Maximum range of the car\'s sensors')
    parser.add_argument("-s", "--sensor_noise_std", default=0.0, type=float, help='Std dev of car\'s sensor noise')
    parser.add_argument("-d", "--gps_noise_dist", default="gaussian", choices=["gaussian", "uniform", "laplace", "cauchy"], help='Type of distribution for GPS sensor noise')
    parser.add_argument("-sd", "--sensor_noise_dist", default="gaussian", choices=["gaussian", "uniform", "laplace", "cauchy"], help='Type of distribution for GPS sensor noise')
    parser.add_argument("-gv", "--gps_noise_var", default=10.0, type=float, help='Variance of gaussian noise for GPS measurement (Kalman filter)')
    parser.add_argument("-gw", "--gps_noise_width", default=20, type=float, help='Width of uniformly random noise for GPS measurement (Kalman filter)')
    parser.add_argument("-f", "--filename", default="plot.png", help="name of image to store plot inside the plots/ directory")
    args = parser.parse_args()

    # Initialize arrays for storing data
    car1_pos_x, car1_pos_y = [], []
    car2_pos_x, car2_pos_y = [], []
    est1_pos_x, est1_pos_y = [], []
    est2_pos_x, est2_pos_y = [], []
    gps1_x, gps1_y = [], []
    gps2_x, gps2_y = [], []

    # Initialize simulator based on filter type
    
    if args.which == "pf":
        max_sensor_range = args.max_sensor_range
        sensor_std = args.sensor_noise_std
        num_particles = args.num_particles
        noise_type = args.sensor_noise_dist

        print(f"Running particle filtering with\n    Num particles = {num_particles}\n    Max sensor range = {max_sensor_range}\n    Sensor noise std = {sensor_std}\n    Noise distribution = {noise_type}")
        
        sim = Simulator(max_sensor_range=max_sensor_range, sensor_std=sensor_std, num_particles=num_particles, noise_type=noise_type)
        sim.toggle_particles()
    elif args.which == "kf":
        noise_type = args.gps_noise_dist
        if noise_type == "gaussian":
            gps_noise_var = args.gps_noise_var
            sim = Simulator(gps_noise_var=gps_noise_var, noise_type=noise_type)
            print(f"Running Kalman filtering with\n    GPS noise dist = {noise_type}\n    GPS gaussian noise var = {gps_noise_var}")
        elif noise_type == "uniform":
            gps_noise_width = args.gps_noise_width
            sim = Simulator(gps_noise_width=gps_noise_width, noise_type=noise_type)
            print(f"Running Kalman filtering with\n    GPS noise dist = {noise_type}\n    GPS uniform noise width = {gps_noise_width}")
        elif noise_type == "laplace":
            gps_noise_var = args.gps_noise_var
            sim = Simulator(gps_noise_var=gps_noise_var, noise_type=noise_type)
            print(f"Running Kalman filtering with\n    GPS noise dist = {noise_type}\n    GPS Laplace scale param = {gps_noise_var}")
        elif noise_type == "cauchy":
            gps_noise_var = args.gps_noise_var
            sim = Simulator(gps_noise_var=gps_noise_var, noise_type=noise_type)
            print(f"Running Kalman filtering with\n    GPS noise dist = {noise_type}\n    GPS Cauchy scale param = {gps_noise_var}")
        else:
            raise ValueError(f"Invalid GPS noise distribution: {noise_type}")
        sim.toggle_kalman()
    else:
        raise ValueError("Invalid filter type")

    sim.toggle_replay()

    # Run simulation and collect data
    segment = 0
    display_progress_every = 25

    for i in range(LAP_TRIM_IDX):
        sim.loop()
        car1_pos_x.append(sim.car1.pos[0])
        car1_pos_y.append(sim.car1.pos[1])

        car2_pos_x.append(sim.car2.pos[0])
        car2_pos_y.append(sim.car2.pos[1])
        
        if args.which == "pf":
            if sim.x_est1 is not None and sim.y_est1 is not None:
                est1_pos_x.append(sim.x_est1)
                est1_pos_y.append(sim.y_est1)
            else:
                est1_pos_x.append(car1_pos_x[-1])  # Use actual position as fallback
                est1_pos_y.append(car1_pos_y[-1])

            if sim.x_est2 is not None and sim.y_est2 is not None:
                est2_pos_x.append(sim.x_est2)
                est2_pos_y.append(sim.y_est2)
            else:
                est2_pos_x.append(car2_pos_x[-1])  # Use actual position as fallback
                est2_pos_y.append(car2_pos_y[-1])
        elif args.which == "kf":
            car2_pos_x.append(sim.car2.pos[0])
            car2_pos_y.append(sim.car2.pos[1])
            
            if sim.kf_state1 is not None:
                est1_pos_x.append(sim.kf_state1[0])
                est1_pos_y.append(sim.kf_state1[1])
            else:
                est1_pos_x.append(car1_pos_x[-1])
                
            if sim.kf_state2 is not None:
                est2_pos_x.append(sim.kf_state2[0])
                est2_pos_y.append(sim.kf_state2[1])
            else:
                est2_pos_x.append(car2_pos_x[-1])
                est2_pos_y.append(car2_pos_y[-1])
                
            if hasattr(sim, 'gps_measurement1') and sim.gps_measurement1 is not None:
                gps1_x.append(sim.gps_measurement1[0])
                gps1_y.append(sim.gps_measurement1[1])
            else:
                gps1_x.append(car1_pos_x[-1])
                gps1_y.append(car1_pos_y[-1])
                
            if hasattr(sim, 'gps_measurement2') and sim.gps_measurement2 is not None:
                gps2_x.append(sim.gps_measurement2[0])
                gps2_y.append(sim.gps_measurement2[1])
            else:
                gps2_x.append(car2_pos_x[-1])
                gps2_y.append(car2_pos_y[-1])

        if int(i % ((LAP_TRIM_IDX - 1) / (100 / display_progress_every))) == 0:
            print(f"{segment * display_progress_every}% complete")
            segment += 1

    # Convert coordinates
    car1_pos_x, car1_pos_y = convert(car1_pos_x, car1_pos_y)
    est1_pos_x, est1_pos_y = convert(est1_pos_x, est1_pos_y)
    if args.which == "kf":
        car2_pos_x, car2_pos_y = convert(car2_pos_x, car2_pos_y)
        est2_pos_x, est2_pos_y = convert(est2_pos_x, est2_pos_y)
        gps1_x, gps1_y = convert(gps1_x, gps1_y)
        gps2_x, gps2_y = convert(gps2_x, gps2_y)

    # Create plots based on filter type
    if args.which == "pf":
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
        # fig.suptitle(f"Particle Filtering N={sim.num_particles}, Range={sim.max_sensor_range}, Noise={sim.sensor_std}")
        
        ax0 = axs[0, 0]
        ax1 = axs[0, 1]
        ax2 = axs[1, 0]
        ax3 = axs[1, 1]

        # Plot car1 position, estimate, and checkpoints
        ax0.set_title("Path and Checkpoints - Car 1")
        ax0.plot(car1_pos_x, car1_pos_y, label="car position", color="black")
        ax0.plot(est1_pos_x, est1_pos_y, label="estimated position", color="blue")
        plot_checkpoints(ax0, sim.checkpoints, sim.car1_checkpoint_reached)


        ax1.set_title("Path and Checkpoints - Car 2")
        ax1.plot(car2_pos_x, car2_pos_y, label="car position", color="black")
        ax1.plot(est2_pos_x, est2_pos_y, label="estimated position", color="blue")
        plot_checkpoints(ax1, sim.checkpoints, sim.car2_checkpoint_reached)

        # Plot checkpoint errors
        ax2.set_title("Error relative to next checkpoint - Car 1")
        ax2.plot(sim.car1_checkpoint_errors)

        ax3.set_title("Error relative to next checkpoint - Car 2")
        ax3.plot(sim.car2_checkpoint_errors)

    elif args.which == "kf":
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
        # if args.gps_noise_dist == "gaussian":
        #     fig.suptitle(f"Kalman Filtering GPS Noise Dist = Gaussian, GPS Noise Var = {sim.gps_noise_var}")
        # elif args.gps_noise_dist == "uniform":
        #     fig.suptitle(f"Kalman Filtering GPS Noise Dist = Uniform, GPS Noise Width = {sim.gps_noise_width}")
            
        ax0 = axs[0, 0]
        ax1 = axs[0, 1]
        ax2 = axs[1, 0]
        ax3 = axs[1, 1]

        # Plot both cars' positions, estimates, and checkpoints
        ax0.set_title("Path and Checkpoints - Car 1")
        ax0.plot(car1_pos_x, car1_pos_y, label="car position", color="black")
        ax0.plot(est1_pos_x, est1_pos_y, label="estimated position", color="blue")
        ax0.plot(gps1_x, gps1_y, label="GPS measurement", color="red")
        plot_checkpoints(ax0, sim.checkpoints, sim.car1_checkpoint_reached)

        ax1.set_title("Path and Checkpoints - Car 2")
        ax1.plot(car2_pos_x, car2_pos_y, label="car position", color="black")
        ax1.plot(est2_pos_x, est2_pos_y, label="estimated position", color="blue")
        ax1.plot(gps2_x, gps2_y, label="GPS measurement", color="red")
        plot_checkpoints(ax1, sim.checkpoints, sim.car2_checkpoint_reached)

        # Plot checkpoint errors
        ax2.set_title("Error relative to next checkpoint - Car 1")
        ax2.plot(sim.car1_checkpoint_errors)

        ax3.set_title("Error relative to next checkpoint - Car 2")
        ax3.plot(sim.car2_checkpoint_errors)

    # Set common properties for all axes
    for ax in fig.get_axes():
        if "Path" in ax.get_title():
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, WORLD_WIDTH)
            ax.set_ylim(0, WORLD_HEIGHT)
            ax.set_xticks(np.arange(0, WORLD_WIDTH, 50))
            ax.set_xticks(np.arange(0, WORLD_WIDTH, 10), minor=True)
            ax.set_yticks(np.arange(0, WORLD_HEIGHT, 50))
            ax.set_yticks(np.arange(0, WORLD_HEIGHT, 10), minor=True)
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)
            ax.legend()
        elif "Error" in ax.get_title():
            ax.set_xlabel("Time")
            ax.set_ylabel("Error")
            ax.set_xlim(0, len(sim.car1_checkpoint_errors))
            ax.set_xticks(np.arange(0, len(sim.car1_checkpoint_errors), 50))
            ax.set_xticks(np.arange(0, len(sim.car1_checkpoint_errors), 10), minor=True)
            if "Car 1" in ax.get_title():
                if len(sim.car1_checkpoint_errors) > 0:
                    max_error = max(sim.car1_checkpoint_errors)
                    ax.set_yticks(np.arange(0, max_error, 100))
                    ax.set_yticks(np.arange(0, max_error, 20), minor=True)
            elif "Car 2" in ax.get_title():
                if len(sim.car2_checkpoint_errors) > 0:
                    max_error = max(sim.car2_checkpoint_errors)
                    ax.set_yticks(np.arange(0, max_error, 100))
                    ax.set_yticks(np.arange(0, max_error, 20), minor=True)
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots/"+args.filename, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()