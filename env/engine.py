import numpy as np
from typing import Tuple, List

VELOCITY_THRESHOLD = 0.01
COLLISION_ITERATIONS = 3
REFERENCE_DT = 0.1


class Object():
    def __init__(self, radius: float, position: Tuple[float, float], velocity: Tuple[float, float],
                 orientation: float, angular_velocity: float, bounceable: bool = False,
                 restitution: float = 1.0, mass: float = 1.0):
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.bounceable = bounceable
        self.restitution = restitution
        self.mass = mass

    def get_speed(self):
        return np.linalg.norm(self.velocity)

    def clip_position(self, min_length: int, max_length: int, min_width: int, max_width: int,
                      goal_min_y: int, goal_max_y: int, goal_depth: int,
                      allow_goal_entry: bool = False):
        x, y = self.position

        if not allow_goal_entry:
            if x - self.radius <= min_length:
                self.position[0] = min_length + self.radius
                if self.bounceable:
                    self.velocity[0] = -self.velocity[0] * self.restitution
            elif x + self.radius >= max_length:
                self.position[0] = max_length - self.radius
                if self.bounceable:
                    self.velocity[0] = -self.velocity[0] * self.restitution
        else:
            in_goal_y = goal_min_y <= y <= goal_max_y

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

        if y - self.radius <= min_width:
            self.position[1] = min_width + self.radius
            if self.bounceable:
                self.velocity[1] = -self.velocity[1] * self.restitution

        elif y + self.radius >= max_width:
            self.position[1] = max_width - self.radius
            if self.bounceable:
                self.velocity[1] = -self.velocity[1] * self.restitution

    def clip_velocity(self, max_speed: float):
        speed = self.get_speed()
        if speed > 1e-8:
            self.velocity = self.velocity / speed * max_speed

    def clip_angular_velocity(self, max_angular_speed: float):
        self.angular_velocity = np.clip(self.angular_velocity, -max_angular_speed, max_angular_speed)

    def apply_friction(self, friction_factor: float, angular_friction_factor: float, dt: float):
        dt_ratio = dt / REFERENCE_DT
        self.velocity *= friction_factor ** dt_ratio
        self.angular_velocity *= angular_friction_factor ** dt_ratio

        if self.get_speed() < VELOCITY_THRESHOLD:
            self.velocity = np.array([0.0, 0.0])
        if abs(self.angular_velocity) < VELOCITY_THRESHOLD:
            self.angular_velocity = 0.0

    def act(self, acceleration: Tuple[float, float], angular_acceleration: float, dt: float,
            max_speed: float, max_angular_speed: float, friction_factor: float,
            angular_friction_factor: float, min_length: int, max_length: int,
            min_width: int, max_width: int, goal_min_y: int, goal_max_y: int,
            goal_depth: int, instant: bool = False, allow_goal_entry: bool = False):

        acceleration = np.array(acceleration)
        if instant:
            self.velocity = self.velocity + acceleration
            self.angular_velocity = self.angular_velocity + angular_acceleration
        else:
            self.velocity = self.velocity + dt * acceleration
            self.angular_velocity = self.angular_velocity + dt * angular_acceleration

        speed = self.get_speed()
        if max_speed and speed > 1e-8 and speed > max_speed:
            self.clip_velocity(max_speed)
        if max_angular_speed:
            self.clip_angular_velocity(max_angular_speed)

        if friction_factor and angular_friction_factor:
            self.apply_friction(friction_factor, angular_friction_factor, dt)

        self.position = self.position + dt * self.velocity
        self.clip_position(min_length, max_length, min_width, max_width, goal_min_y, goal_max_y, goal_depth, allow_goal_entry)

        self.orientation = self.orientation + dt * self.angular_velocity
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

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

    for _ in range(COLLISION_ITERATIONS):
        any_collision = False

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                obj1 = objects[i]
                obj2 = objects[j]

                delta_pos = obj1.position - obj2.position
                min_dist = obj1.radius + obj2.radius
                dist = distance(obj1, obj2)

                if dist < min_dist:
                    any_collision = True
                    overlap = min_dist - dist
                    direction = delta_pos / max(dist, 1e-8)

                    obj1.position += direction * (overlap / 2)
                    obj2.position -= direction * (overlap / 2)

                    relative_velocity = obj1.velocity - obj2.velocity
                    velocity_along_normal = np.dot(relative_velocity, direction)

                    if velocity_along_normal > 0:
                        continue

                    combined_restitution = min(obj1.restitution, obj2.restitution)
                    impulse = -(1 + combined_restitution) * velocity_along_normal

                    obj1.velocity += impulse * direction
                    obj2.velocity -= impulse * direction

        if not any_collision:
            break

    return objects
