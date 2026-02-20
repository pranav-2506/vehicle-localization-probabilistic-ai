import sys
import argparse
import math
from tkinter import *
from PIL import Image, ImageTk
from utils import load_image
from simulator import Simulator, WORLD_WIDTH, WORLD_HEIGHT
from racetrack import RaceTrack, Contour, Horizontals, load_racetrack

MAX_COUNT_SINCE = 5

class App(Tk):
    TICK_RATE = 40
    def change_car_color(self, image, target_color):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        

        new_image = Image.new('RGBA', image.size)
        
        pixels = image.load()
        new_pixels = new_image.load()
      
        for i in range(image.width):
            for j in range(image.height):
                r, g, b, a = pixels[i, j]
                if b > r and b > g:
                    brightness = (r + g + b) / 3
                    new_r = int(target_color[0] * (brightness / 255))
                    new_g = int(target_color[1] * (brightness / 255))
                    new_b = int(target_color[2] * (brightness / 255))
                    new_pixels[i, j] = (new_r, new_g, new_b, a)
                else:
       
                    new_pixels[i, j] = (r, g, b, a)
        
        return new_image
    def __init__(self, max_sensor_range=50, sensor_std=0.0, num_particles=50, gps_noise_var=10.0, gps_noise_width=20, noise_type="gaussian"):
        Tk.__init__(self, None, baseName=None,
                    className='Tk', useTk=1, sync=0, use=None)
        max_width = self.winfo_screenwidth() * 0.85
        max_height = self.winfo_screenheight() * 0.85
        scale_width = max_width / WORLD_WIDTH
        scale_height = max_height / WORLD_HEIGHT
        self.scale = min(min(scale_width, scale_height), 1.0)
        
        self.draw_occupancy = False
        self.draw_particles = True
        self.__canvas = Canvas(self, width=WORLD_WIDTH * self.scale, height=WORLD_HEIGHT * self.scale)
        self.__canvas.pack()
        self.__canvas.configure(background="red")
        self.title("Simulator")
        
        self.bind("<KeyPress>", self.keydown)
        self.bind("<KeyRelease>", self.keyup)
        self.history_chars = []
        _, self.bg = load_image("data/track.png", self.scale)
        
        img_car_blue, _ = load_image("data/car_blue.png", self.scale)
        img_car_red = self.change_car_color(img_car_blue, target_color=(255, 0, 0))

        self.car_blue_imgs = []
        self.car_red_imgs = []
        for i in range(-180,180):
            imgtk_blue = ImageTk.PhotoImage(img_car_blue.rotate(i))
            imgtk_red = ImageTk.PhotoImage(img_car_red.rotate(i))
            self.car_blue_imgs.append(imgtk_blue)
            self.car_red_imgs.append(imgtk_red)

        img_green_arrow, _ = load_image("data/green_arrow.png", self.scale)
        self.green_arrow_imgs = []
        for i in range(-180,180):
            imgtk = ImageTk.PhotoImage(img_green_arrow.rotate(i))
            self.green_arrow_imgs.append(imgtk)

        self.count_since = 0
        self.max_count_since = MAX_COUNT_SINCE

        self.simulator = Simulator(max_sensor_range=max_sensor_range, sensor_std=sensor_std, num_particles=num_particles, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width, noise_type=noise_type)
    
        self.car1_label = Label(self, text="Car 1: Lap 0")
        self.car1_label.pack()
        self.car2_label = Label(self, text="Car 2: Lap 0")
        self.car2_label.pack()
        
        self.noise_label = Label(self, text=f"Noise Distribution: {noise_type}", font=("Arial", 10, "bold"))
        self.noise_label.pack()
    
    def keyup(self, e):
        if e.keysym in self.history_chars:
            self.history_chars.pop(self.history_chars.index(e.keysym))

    def keydown(self, e):
        if not e.keysym in self.history_chars:
            self.history_chars.append(e.keysym)

    def process_input(self):
        self.count_since += 1
        self.count_since = min(self.count_since, self.max_count_since)
        if not self.simulator.replaying:
            if "Up" in self.history_chars:
                self.simulator.car1.throttle_press()
            if "Down" in self.history_chars:
                self.simulator.car1.brake_press()
            if "Left" in self.history_chars:
                self.simulator.car1.steer("left")
            if "Right" in self.history_chars:
                self.simulator.car1.steer("right")
            
 
            if "w" in self.history_chars:
                self.simulator.car2.throttle_press()
            if "s" in self.history_chars:
                self.simulator.car2.brake_press()
            if "a" in self.history_chars:
                self.simulator.car2.steer("left")
            if "d" in self.history_chars:
                self.simulator.car2.steer("right")

        if "p" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.simulator.toggle_particles()
                self.count_since = 0
        if "k" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.simulator.toggle_kalman()
                self.count_since = 0
        if "o" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.draw_occupancy = not self.draw_occupancy
                self.count_since = 0
        if "r" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.draw_particles = not self.draw_particles
                self.count_since = 0
        if "g" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.simulator.toggle_gps_noise_dist()
                self.count_since = 0
                print("GPS noise dist is now {}".format(self.simulator.gps_noise_dist))
    
                self.noise_label.config(text=f"Noise Distribution: {self.simulator.gps_noise_dist}")

    def show_game_over_message(self):

        game_over_window = Toplevel(self)
        game_over_window.title("Game Over")
        game_over_window.geometry("300x100")
        
        label = Label(game_over_window, text=f"Game Over!\n{self.simulator.winner} wins!", font=("Arial", 16))
        label.pack(expand=True)
    
    def __loop(self):

        if not self.simulator.loop():
        
            self.show_game_over_message()
            self.after(3000, self.destroy) 
            return

        self.after(App.TICK_RATE, self.__loop)

        self.process_input()
        self.simulator.loop()
        self.__canvas.delete(ALL)
        self.__canvas.create_image(0,0,image=self.bg,anchor=NW,tag="bg")

        car1 = self.simulator.car1
        car2 = self.simulator.car2
        racetrack = self.simulator.racetrack
        
        car1_angle = int(math.degrees(math.atan2(-car1.orient[1], car1.orient[0])))
        car2_angle = int(math.degrees(math.atan2(-car2.orient[1], car2.orient[0])))
        self.__canvas.create_image(car1.pos[0] * self.scale, car1.pos[1] * self.scale, image=self.car_blue_imgs[car1_angle + 180])
        self.__canvas.create_image(car2.pos[0] * self.scale, car2.pos[1] * self.scale, image=self.car_red_imgs[car2_angle + 180])


        if self.draw_occupancy:
            for i in range(80):
                self.__canvas.create_line(
                    0, 10 * i * self.scale, 1400 * self.scale, 10 * i * self.scale, fill="gray")
            for i in range(140):
                self.__canvas.create_line(
                    10 * i * self.scale, 0, 10 * i * self.scale, 800 * self.scale, fill="gray")
            for i in range(140):
                for j in range(80):
                    if racetrack.occupancy[i,j] != 0:
                        self.__canvas.create_rectangle(i * 10 * self.scale, j * 10 * self.scale, (i * 10 + 10) * self.scale, (j * 10 + 10) * self.scale, fill="black")

        sensor_color = "red"
        sensor_width = 2
        if self.draw_occupancy:
            for car in [car1, car2]:
                dists = car.sensor_dists
                self.__canvas.create_line(car.pos[0] * self.scale, car.pos[1] * self.scale, car.pos[0] * self.scale, (car.pos[1]-dists[0]) * self.scale, fill=sensor_color, width=sensor_width)
                self.__canvas.create_line(car.pos[0] * self.scale, car.pos[1] * self.scale, car.pos[0] * self.scale, (car.pos[1]+dists[1]) * self.scale, fill=sensor_color, width=sensor_width)
                self.__canvas.create_line(car.pos[0] * self.scale, car.pos[1] * self.scale, (car.pos[0]-dists[2]) * self.scale, car.pos[1] * self.scale, fill=sensor_color, width=sensor_width)
                self.__canvas.create_line(car.pos[0] * self.scale, car.pos[1] * self.scale, (car.pos[0]+dists[3]) * self.scale, car.pos[1] * self.scale, fill=sensor_color, width=sensor_width)
        
        if self.draw_particles:
            if self.simulator.do_particle_filtering:
                for p in self.simulator.particle_filter1.particles:
                    self.__canvas.create_oval((p.pos[0]-2) * self.scale,(p.pos[1]-2) * self.scale,(p.pos[0]+2) * self.scale,(p.pos[1]+2) * self.scale,fill="red")
                for p in self.simulator.particle_filter2.particles:
                    self.__canvas.create_oval((p.pos[0]-2) * self.scale,(p.pos[1]-2) * self.scale,(p.pos[0]+2) * self.scale,(p.pos[1]+2) * self.scale,fill="blue")

        if self.simulator.do_particle_filtering:
            angle_est1 = int(math.degrees(math.atan2(-self.simulator.orient_est1[1], self.simulator.orient_est1[0])))
            self.__canvas.create_image(self.simulator.x_est1 * self.scale, self.simulator.y_est1 * self.scale, image=self.green_arrow_imgs[angle_est1 + 180])
        
            angle_est2 = int(math.degrees(math.atan2(-self.simulator.orient_est2[1], self.simulator.orient_est2[0])))
            self.__canvas.create_image(self.simulator.x_est2 * self.scale, self.simulator.y_est2 * self.scale, image=self.green_arrow_imgs[angle_est2 + 180])
        
        if self.simulator.do_kalman_filtering:
            for car, kf_state in [(car1, self.simulator.kf_state1), (car2, self.simulator.kf_state2)]:
                measx = car.gps_measurement[0]
                measy = car.gps_measurement[1]
                self.__canvas.create_oval((measx-4) * self.scale,(measy-4) * self.scale,(measx+4) * self.scale,(measy+4) * self.scale,fill="red")
                kfx = kf_state[0]
                kfy = kf_state[1]
                self.__canvas.create_oval((kfx-4) * self.scale,(kfy-4) * self.scale,(kfx+4) * self.scale,(kfy+4) * self.scale,fill="lime green")
        
        for checkpoint in self.simulator.checkpoints:
            self.__canvas.create_oval(
                (checkpoint[0] - 5) * self.scale,
                (checkpoint[1] - 5) * self.scale,
                (checkpoint[0] + 5) * self.scale,
                (checkpoint[1] + 5) * self.scale,
                fill="yellow"
            )

        self.car1_label.config(text=f"Car 1: Lap {self.simulator.car1_laps}")
        self.car2_label.config(text=f"Car 2: Lap {self.simulator.car2_laps}")
        
    def mainloop(self, n=0):
        self.__loop()
        Tk.mainloop(self, n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_particles", default=50, type=int, help='Number of particles for particle filtering')
    parser.add_argument("-m", "--max_sensor_range", default=50, type=int, help='Maximum range of the car\'s sensors')
    parser.add_argument("-s", "--sensor_noise_std", default=0.0, type=float, help='Std dev of car\'s sensor noise')
    parser.add_argument("-gv", "--gps_noise_var", default=10.0, type=float, help='Variance of gaussian noise for GPS measurement (Kalman filter)')
    parser.add_argument("-gw", "--gps_noise_width", default=20, type=float, help='Width of uniformly random noise for GPS measurement (Kalman filter)')
    parser.add_argument("-nt", "--noise_type", default="gaussian", choices=["gaussian", "uniform", "laplace", "cauchy"], help='Noise distribution type (gaussian, uniform, laplace, cauchy)')
    args = parser.parse_args()

    max_sensor_range = args.max_sensor_range
    sensor_std = args.sensor_noise_std
    num_particles = args.num_particles
    gps_noise_var = args.gps_noise_var
    gps_noise_width = args.gps_noise_width
    noise_type = args.noise_type

    print("Running GUI with\n    Num particles = {}\n    Max sensor range = {}\n    Sensor noise std = {}\n    GPS gaussian noise var={}\n    GPS uniform noise width={}\n    Noise type = {}".format(num_particles, max_sensor_range, sensor_std, gps_noise_var, gps_noise_width, noise_type))

    app = App(max_sensor_range=max_sensor_range, sensor_std=sensor_std, num_particles=num_particles, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width, noise_type=noise_type)
    app.mainloop()

if __name__ == "__main__":
    main()
