import numpy as np
from particle_filter import ParticleFilter, Particle, weight_gaussian_kernel
from utils import add_noise as utils_add_noise, add_noise_laplace, add_noise_cauchy


class ParticleExtra(Particle):

    
    def add_noise(self, std_pos=1.0, std_orient=1.0, noise_type="gaussian"):

        if noise_type == "gaussian":
          
            noise_func = utils_add_noise
            param_pos = std_pos
            param_orient = std_orient
        elif noise_type == "laplace":
           
            noise_func = add_noise_laplace
            param_pos = std_pos * 1.2
            param_orient = std_orient * 1.2
        elif noise_type == "cauchy":
           
            noise_func = add_noise_cauchy
            param_pos = std_pos * 0.5
            param_orient = std_orient * 0.5
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Apply noise to position (reusing the pattern from base Particle)
        self.pos[0] = noise_func(x=self.pos[0], scale=param_pos)
        self.pos[1] = noise_func(x=self.pos[1], scale=param_pos)
        
        # Apply noise to orientation and normalize (reusing the pattern from base Particle)
        while True:
            self.orient[0] = noise_func(x=self.orient[0], scale=param_orient)
            self.orient[1] = noise_func(x=self.orient[1], scale=param_orient)
            if np.linalg.norm(self.orient) >= 1e-8:
                break
        self.orient = self.orient / np.linalg.norm(self.orient)


class ParticleFilterExtra(ParticleFilter):
    
    def __init__(self, num_particles, minx, maxx, miny, maxy, noise_type="gaussian"):
        
        super().__init__(num_particles, minx, maxx, miny, maxy)
        self.noise_type = noise_type
    
    def transition_sample(self, particle, delta_angle, speed):
     
        new_particle = None

        # new_particle.add_noise(std_pos=1.0, std_orient=0.1, noise_type=self.noise_type)
        
        raise NotImplementedError("implementation from particle_filter.py")
        
      
        return new_particle
    
    def compute_prenorm_weight(self, particle, sensor, max_sensor_range, sensor_std, evidence):
        
        weight = None
    
        
        raise NotImplementedError("compute_prenorm_weight implementation from particle_filter.py")
        
    
        return weight

def weight_laplace_kernel(x1, x2, scale=500):

    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return np.exp(-distance / scale)


def weight_cauchy_kernel(x1, x2, scale=500):
    
    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return 1.0 / (1.0 + (distance / scale) ** 2)
