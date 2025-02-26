import numpy as np
from typing import Tuple

class Object():
    def __init__(self, radius: float, position: Tuple[float, float], velocity: Tuple[float, float], rotation: float, angular_velocity: float):
        self.radius = radius
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.rotation = rotation
        self.angular_velocity = angular_velocity
        
    def get_speed(self):
        return np.linalg.norm(self.velocity)
    
    def clip_position(self, min_length: int, max_length: int, min_width: int, max_width: int):
        self.position[0] = np.clip(self.position[0], min_length, max_length)
        self.position[1] = np.clip(self.position[1], min_width, max_width)
    
    def clip_velocity(self, max_speed: float):
        self.velocity = self.velocity / self.get_speed() * max_speed
        
    def clip_angular_velocity(self, max_angular_speed: float):
        self.angular_velocity = np.clip(self.angular_velocity, -max_angular_speed, max_angular_speed)
        
    def apply_friction(self, friction_factor: float, angular_friction_factor: float):
        self.velocity = self.velocity * friction_factor
        self.angular_velocity = self.angular_velocity * angular_friction_factor
        
    def act(self, acceleration: Tuple[float, float], angular_acceleration: float, dt: float, max_speed: float, max_angular_speed: float, friction_factor: float, angular_friction_factor: float, min_length: int, max_length: int, min_width: int, max_width: int):
        
        acceleration = np.array(acceleration)
        self.velocity = self.velocity + dt * acceleration
        self.angular_velocity = self.angular_velocity + dt * angular_acceleration
        speed = self.get_speed()
        if max_speed and speed != 0 and speed > max_speed:
            self.clip_velocity(max_speed)
        if max_angular_speed:
            self.clip_angular_velocity(max_angular_speed)
        if friction_factor and angular_friction_factor:
            self.apply_friction(friction_factor, angular_friction_factor)
        
        self.position = self.position + dt * self.velocity
        self.clip_position(min_length, max_length, min_width, max_width)
        self.rotation = self.rotation + dt * self.angular_velocity
        
        
    def __str__(self):
        return f"Object(radius={self.radius}, position={self.position}, velocity={self.velocity}, rotation={self.rotation}, angular_velocity={self.angular_velocity})"
    
class Ball(Object):
    def __init__(self, object: Object):
        self.object = object
        
    def __str__(self):
        return f"Ball({self.object})"
        
class Player(Object):
    def __init__(self, object: Object, team: int):
        self.object = object
        self.team = team
        
    def __str__(self):
        return f"Player({self.object}, team={self.team})"
        
def distance(obj1: Object, obj2: Object):
    return np.linalg.norm(obj1.position - obj2.position)