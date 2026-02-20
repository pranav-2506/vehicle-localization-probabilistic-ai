""" Particle filtering """

import random
import numpy as np
import bisect
import copy
from utils import add_noise as utils_add_noise
class Particle:

    def __init__(self, pos, orient, weight=1.0):
    
        self.pos = pos
        self.orient = orient
        self.weight = weight
    
    def add_noise(self, std_pos=1.0, std_orient=1.0):
    
        self.pos[0] = utils_add_noise(x=self.pos[0], std=std_pos)
        self.pos[1] = utils_add_noise(x=self.pos[1], std=std_pos)
        while True:
            self.orient[0] = utils_add_noise(x=self.orient[0], std=std_orient)
            self.orient[1] = utils_add_noise(x=self.orient[1], std=std_orient)
            if np.linalg.norm(self.orient) >= 1e-8:
                break
        self.orient = self.orient / np.linalg.norm(self.orient)

class ParticleFilter:
 
    def __init__(self, num_particles, minx, maxx, miny, maxy):
    
        self.num_particles = num_particles
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.particles = self.initialize_particles()
        
    def initialize_particles(self):
     
        particles = []

       
        for _ in range(self.num_particles):
            x = np.random.uniform(self.minx, self.maxx)
            y = np.random.uniform(self.miny, self.maxy)
            pos = np.array([x, y])
            theta = np.random.uniform(0, 2 * np.pi)
            orient = np.array([np.cos(theta), np.sin(theta)])
            particles.append(Particle(pos, orient, weight=1.0))

        return particles

    
    def filtering_and_estimation(self, sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed):
    

        # run filtering step to update particles
        self.particles = self.filtering(sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed)

        for p in self.particles:
            self.fix_particle(p)

        # compute estimated position, angle
        x_est, y_est, orient_est = estimate_pose(self.particles)

        return x_est, y_est, orient_est
    
    def filtering(self, sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed):
    
        new_particles = []

       
        temp_particles = []
        for p in self.particles:
            p_prime = self.transition_sample(p, delta_angle, speed)
            w = self.compute_prenorm_weight(p_prime, sensor, max_sensor_range, sensor_std, evidence)
            p_prime.weight = w
            temp_particles.append(p_prime)
        normalize_weights(temp_particles)
        new_particles = self.weighted_sample_w_replacement(temp_particles)
  

        return new_particles
    
    def compute_prenorm_weight(self, particle, sensor, max_sensor_range, sensor_std, evidence):
 
        weight = None
        
        x = float(particle.pos[0])
        y = float(particle.pos[1])
        predicted_readings = sensor(x, y, max_sensor_range)
        weight = weight_gaussian_kernel(predicted_readings, evidence, std=50)
        if np.isnan(weight) or np.isinf(weight):
            weight = 0.0
    
        return weight

    def transition_sample(self, particle, delta_angle, speed):
   
        new_particle = None
    
        cos_a = np.cos(delta_angle)
        sin_a = np.sin(delta_angle)
        old_x = particle.orient[0]
        old_y = particle.orient[1]
        new_orient_x = old_x * cos_a - old_y * sin_a
        new_orient_y = old_x * sin_a + old_y * cos_a
        new_orient = np.array([new_orient_x, new_orient_y])
        new_orient = new_orient / np.linalg.norm(new_orient)
        new_pos = particle.pos + new_orient * speed
        new_particle = Particle(new_pos.copy(), new_orient.copy(), particle.weight)
        new_particle.add_noise(std_pos=0.5, std_orient=0.08)
     
        return new_particle
    
    def fix_particle(self, particle):
    
        x = particle.pos[0]
        y = particle.pos[1]
        particle.pos[0] = max(min(x,self.maxx),self.minx)
        particle.pos[1] = max(min(y,self.maxy),self.miny)
        return particle
    
    def weighted_sample_w_replacement(self, particles):
        new_particles = []

        distribution = WeightedDistribution(particles=particles)

        for _ in range(len(particles)):
            particle = distribution.random_select()
            if particle is None:
                pos = np.array([np.random.uniform(self.minx, self.maxx), np.random.uniform(self.miny, self.maxy)])
                orient = np.array([random.random() - 0.5, random.random() - 0.5])
                orient = orient / np.linalg.norm(orient)
                new_particles.append(Particle(pos, orient))
            else:
                p = Particle(copy.deepcopy(particle.pos), copy.deepcopy(particle.orient))
                new_particles.append(p)
        
        return new_particles

def weight_gaussian_kernel(x1, x2, std = 500):
 
    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return np.exp(-distance ** 2 / (2 * std))

def normalize_weights(particles):
   
    weight_total = 0
    for p in particles:
        weight_total += p.weight

    if weight_total == 0:
        weight_total = 1e-8

    for p in particles:
        p.weight /= weight_total

class WeightedDistribution(object):

    def __init__(self, particles):
        
        accum = 0.0
        self.particles = particles
        self.distribution = list()
        for particle in self.particles:
            accum += particle.weight
            self.distribution.append(accum)

    def random_select(self):

        try:
            return self.particles[bisect.bisect_left(self.distribution, np.random.uniform(0, 1))]
        except IndexError:
          
            return None

def estimate_pose(particles):
    pos_accum = np.array([0,0])
    orient_accum = np.array([0,0])
    weight_accum = 0.0
    for p in particles:
        weight_accum += p.weight
        pos_accum = pos_accum + p.pos * p.weight
        orient_accum = orient_accum + p.orient * p.weight
    if weight_accum != 0:
        x_est = pos_accum[0] / weight_accum
        y_est = pos_accum[1] / weight_accum
        orient_est = orient_accum / weight_accum
        return x_est, y_est, orient_est
    else:
        raise ValueError
