# !python
import numpy as np
import signalz

class InputGenerator:
    
    def __init__(self, start_time, end_time, num_time_steps):
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps
        
    def generate_sin(self, amplitude=1.0):
        #return np.sin( np.linspace(self.start_time, self.end_time, self.num_time_steps) ) * amplitude
        return np.linspace(self.start_time, self.end_time, self.num_time_steps) * amplitude