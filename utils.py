import math
import numpy as np
import sys
from PIL import Image, ImageTk

def length(x):
    return math.sqrt(x[0]**2 + x[1]**2)

def angle_bw(x, y):
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    if normx <= 1e-8 or normy <= 1e-8:
        return 0
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    return -math.degrees(math.asin((x[0] * y[1] - x[1] * y[0])/(length(x)*length(y))))

def add_noise(x, std):
    return x + np.random.normal(0, std)

def add_noise_laplace(x, scale):
   
       raise NotImplementedError
   
def add_noise_cauchy(x, scale):
    
    
    raise NotImplementedError
    
def load_image(path, scale):
    try:
        img = Image.open(path)
        new_width = int(img.width * float(scale))
        new_height = int(img.height * float(scale))
        img = img.resize((new_width, new_height), Image.LANCZOS)
        return img, ImageTk.PhotoImage(img)
    except IOError as e:
        print(e)
        sys.exit(1)
