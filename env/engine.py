import numpy as np
from typing import Tuple, List

class Object():
    def __init__(self, radius: float, position: Tuple[float, float], velocity: Tuple[float, float], orientation: float, angular_velocity: float, bounceable: bool = False, restitution: float = 1.0):
        self.radius = radius
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.bounceable = bounceable
        self.restitution = restitution
        
    def get_speed(self):
        return np.linalg.norm(self.velocity)
        
    def clip_position(self, min_length: int, max_length: int, min_width: int, max_width: int, 
                    goal_min_y: int, goal_max_y: int, goal_depth: int):

        x, y = self.position

        # Check if object is inside the goal area (y-range)
        in_goal_y = goal_min_y <= y <= goal_max_y     

        # Left wall bounce
        if x - self.radius <= min_length:
            if in_goal_y:
                if x - self.radius <= -goal_depth:
                    self.position[0] = -goal_depth + self.radius
                    if self.bounceable:
                        self.velocity[0] = -self.velocity[0] * self.restitution
                if y - self.radius <= goal_min_y:
                    self.position[1] = goal_min_y + self.radius
                    if self.bounceable:
                        self.velocity[1] = -self.velocity[1] * self.restitution
                elif y + self.radius >= goal_max_y:
                    self.position[1] = goal_max_y - self.radius
                    if self.bounceable:
                        self.velocity[1] = -self.velocity[1] * self.restitution
            else:
                self.position[0] = min_length + self.radius
                if self.bounceable:
                    self.velocity[0] = -self.velocity[0] * self.restitution

        # Right wall bounce
        elif x + self.radius >= max_length:
            if in_goal_y:
                if x + self.radius >= max_length + goal_depth:
                    self.position[0] = max_length + goal_depth - self.radius
                    if self.bounceable:
                        self.velocity[0] = -self.velocity[0] * self.restitution
                if y - self.radius <= goal_min_y:
                    self.position[1] = goal_min_y + self.radius
                    if self.bounceable:
                        self.velocity[1] = -self.velocity[1] * self.restitution
                elif y + self.radius >= goal_max_y:
                    self.position[1] = goal_max_y - self.radius
                    if self.bounceable:
                        self.velocity[1] = -self.velocity[1] * self.restitution
            else:
                self.position[0] = max_length - self.radius
                if self.bounceable:
                    self.velocity[0] = -self.velocity[0] * self.restitution

        # Top wall bounce
        if y - self.radius <= min_width:
            self.position[1] = min_width + self.radius
            if self.bounceable:
                self.velocity[1] = -self.velocity[1] * self.restitution

        # Bottom wall bounce
        elif y + self.radius >= max_width:
            self.position[1] = max_width - self.radius
            if self.bounceable:
                self.velocity[1] = -self.velocity[1] * self.restitution

    def clip_velocity(self, max_speed: float):
        self.velocity = self.velocity / self.get_speed() * max_speed
        
    def clip_angular_velocity(self, max_angular_speed: float):
        self.angular_velocity = np.clip(self.angular_velocity, -max_angular_speed, max_angular_speed)
        
    def apply_friction(self, friction_factor: float, angular_friction_factor: float):
        self.velocity = self.velocity * friction_factor
        self.angular_velocity = self.angular_velocity * angular_friction_factor
        
    def act(self, acceleration: Tuple[float, float], angular_acceleration: float, dt: float, max_speed: float, max_angular_speed: float, friction_factor: float, angular_friction_factor: float, min_length: int, max_length: int, min_width: int, max_width: int, goal_min_y: int, goal_max_y: int, goal_depth: int, instant: bool = False):
        
        acceleration = np.array(acceleration)
        if instant:
            self.velocity = self.velocity + acceleration
            self.angular_velocity = self.angular_velocity + angular_acceleration
        else:
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
        self.clip_position(min_length, max_length, min_width, max_width, goal_min_y, goal_max_y, goal_depth)
        self.orientation = self.orientation + dt * self.angular_velocity
        
        
    def __str__(self):
        return f"Object(radius={self.radius}, position={self.position}, velocity={self.velocity}, orientation={self.orientation}, angular_velocity={self.angular_velocity})"
    
class Ball():
    def __init__(self, object: Object):
        self.object = object
        
    def __str__(self):
        return f"Ball({self.object})"
        
class Player():
    def __init__(self, object: Object, team: int):
        self.object = object
        self.team = team
        
    def __str__(self):
        return f"Player({self.object}, team={self.team})"
        
def distance(obj1: Object, obj2: Object):
    return np.linalg.norm(obj1.position - obj2.position)

def resolve_collisions(objects: List[Object]):
    num_objects = len(objects)
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1 = objects[i]
            obj2 = objects[j]

            # Compute distance between objects
            delta_pos = obj1.position - obj2.position
            min_dist = obj1.radius + obj2.radius  # Collision threshold
            dist = distance(obj1, obj2)

            if dist < min_dist:  # Collision detected
                # Compute overlap amount
                overlap = min_dist - dist
                direction = delta_pos / dist  # Normalize direction

                # Separate objects to prevent overlap
                obj1.position += direction * (overlap / 2)
                obj2.position -= direction * (overlap / 2)

                # Compute relative velocity along the collision normal
                relative_velocity = obj1.velocity - obj2.velocity
                velocity_along_normal = np.dot(relative_velocity, direction)

                if velocity_along_normal > 0:
                    continue  # Objects are already moving apart, skip resolution

                # Compute impulse for each object
                impulse1 = -(1 + obj1.restitution) * velocity_along_normal
                impulse2 = -(1 + obj2.restitution) * velocity_along_normal

                # Apply impulse to velocities (relative to each object's restitution)
                obj1.velocity += impulse1 * direction
                obj2.velocity -= impulse2 * direction
    return objects
