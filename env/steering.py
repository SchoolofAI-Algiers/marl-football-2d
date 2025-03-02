import math
import numpy as np
import env.config as config
from env.models import MovementInput, MovementOutput

class SteeringBehaviors:
    '''General functions structure:
    Input : 
    - movementIput containing: current position , velocity,acceleration
    - target position
    Output: 
    - Movement OUtput
    '''

    @staticmethod
    def velocityMatch(player: MovementInput, target: MovementInput) -> MovementOutput:
        acceleration = target.velocity - player.velocity
        return MovementOutput(linear=acceleration, angular=0.0)

    @staticmethod
    def positionMatch(player: MovementInput, target: MovementInput) -> MovementOutput:
        acceleration = target.position - player.position
        return MovementOutput(linear=acceleration, angular=0.0)

    @staticmethod
    def orientationMatch(player: MovementInput, target: MovementInput) -> MovementOutput:
        rotation = target.rotation - player.rotation
        rotation = (rotation + math.pi) % (2 * math.pi) - math.pi  # Map to (-π, π)
        rotation_size = abs(rotation)

        if rotation_size < config.TARGET_RADIUS:
            return MovementOutput(linear=np.array((0.0, 0.0)), angular=0.0)

        target_rotation = (
            config.PLAYER_MAX_ROTATION
            if rotation_size > config.SLOW_RADIUS
            else config.PLAYER_MAX_ROTATION * (rotation_size / config.SLOW_RADIUS)
        )
        target_rotation *= rotation / rotation_size

        angular_acceleration = target_rotation - player.angular_velocity
        return MovementOutput(linear=np.array((0.0, 0.0)), angular=angular_acceleration)

    @staticmethod
    def seek(player: MovementInput, target: MovementInput) -> MovementOutput:
        print(f'seek target: {target}')
        direction = target.position - player.position
        if np.linalg.norm(direction) > 0:
            direction = (direction / np.linalg.norm(direction)) * config.PLAYER_MAX_SPEED
        acceleration = direction - player.velocity
        return MovementOutput(linear=acceleration, angular=0.0)

    @staticmethod
    def flee(player: MovementInput, target: MovementInput) -> MovementOutput:
        direction = player.position - target.position
        if np.linalg.norm(direction) > 0:
            direction = (direction / np.linalg.norm(direction)) * config.PLAYER_MAX_SPEED
        acceleration = direction - player.velocity
        return MovementOutput(linear=acceleration, angular=0.0)

    @staticmethod
    def arrive(player: MovementInput, target: MovementInput) -> MovementOutput:
        direction = target.position - player.position
        distance = np.linalg.norm(direction)

        if distance < config.TARGET_RADIUS:
            return MovementOutput(linear=np.array((0.0, 0.0)), angular=0.0)

        direction = direction / distance if distance > 0 else direction
        target_speed = (
            config.PLAYER_MAX_SPEED
            if distance > config.SLOW_RADIUS
            else config.PLAYER_MAX_SPEED * (distance / config.SLOW_RADIUS)
        )
        desired_velocity = direction * target_speed
        acceleration = (desired_velocity - player.velocity) / 0.1
        return MovementOutput(linear=acceleration, angular=0.0)

    @staticmethod
    def pursue(player: MovementInput, target: MovementInput) -> MovementOutput:
        direction = target.position - player.position
        distance = np.linalg.norm(direction)
        speed = np.linalg.norm(player.velocity)

        prediction = (
            config.MAX_PREDICTION_TIME
            if speed <= distance / config.MAX_PREDICTION_TIME
            else distance / speed
        )
        predicted_position = target.position + target.velocity * prediction
        return SteeringBehaviors.seek(player, MovementInput(position=predicted_position, velocity=np.array((0.0, 0.0)), rotation=0.0, angular_velocity=0.0))

    @staticmethod
    def evade(player: MovementInput, target: MovementInput) -> MovementOutput:
        direction = player.position - target.position
        distance = np.linalg.norm(direction)
        speed = np.linalg.norm(player.velocity)

        prediction = (
            config.MAX_PREDICTION_TIME
            if speed <= distance / config.MAX_PREDICTION_TIME
            else distance / speed
        )
        predicted_position = target.position + target.velocity * prediction
        return SteeringBehaviors.flee(player, MovementInput(position=predicted_position, velocity=np.array((0.0, 0.0)), rotation=0.0, angular_velocity=0.0))

    @staticmethod
    def separation(player: MovementInput, neighbors: list[MovementInput]) -> MovementOutput:
        steering = np.array((0.0, 0.0))
        for neighbor in neighbors:
            direction = player.position - neighbor.object.position
            distance = np.linalg.norm(direction)

            if 0 < distance < config.STOP_THRESHOLD:
                strength = min(config.DECAY_COEFFICIENT / (distance ** 2), config.PLAYER_MAX_ACCELERATION)
                steering += (direction / distance) * strength

        return MovementOutput(linear=steering, angular=0.0)
    

    @staticmethod
    def collision_avoidance(movement_input: MovementInput, target: MovementInput) -> MovementOutput:
        shortest_time = float('inf')
        first_target = None
        first_relative_pos = None
        first_relative_vel = None
        first_min_separation = None
        first_distance = None
        
        relative_pos = target.position - movement_input.position
        relative_vel = target.velocity - movement_input.velocity
        print(f"relative pos: {relative_pos}")
        print(f"relative_vel: {relative_vel}")
        print(f"target: {target}")
        relative_speed = np.linalg.norm(relative_vel)
        if relative_speed == 0:
            return MovementOutput(linear=np.array((0.0, 0.0)), angular=0.0)
        
        time_to_collision = np.dot(relative_pos, relative_vel) / (relative_speed ** 2)
        distance = np.linalg.norm(relative_pos)
        min_separation = distance - relative_speed * time_to_collision
        
        if min_separation > 2 * config.SLOW_RADIUS:
            return MovementOutput(linear=np.array((0.0, 0.0)), angular=0.0)
        
        if 0 < time_to_collision < shortest_time:
            shortest_time = time_to_collision
            first_target = target
            first_relative_pos = relative_pos
            first_relative_vel = relative_vel
            first_min_separation = min_separation
            first_distance = distance
        
        if first_target is None:
            return MovementOutput(linear=np.array((0.0, 0.0)), angular=0.0)
        
        relative_pos = first_relative_pos + first_relative_vel * shortest_time if first_min_separation > 0 and first_distance >= 2 * config.SLOW_RADIUS else first_target.position - movement_input.position
        relative_pos = (relative_pos / np.linalg.norm(relative_pos)) * config.PLAYER_MAX_ACCELERATION
        return MovementOutput(linear=relative_pos, angular=0.0)
    
    @staticmethod
    def wander(player: MovementInput) -> MovementOutput:
        """
        Produces a wandering movement by generating small random variations in direction.
        
    
            player (MovementInput): The player executing the wander behavior.
            wander_radius (float): The radius of the wander circle.
            wander_distance (float): The distance of the wander circle from the player.
            jitter (float): The randomness factor affecting direction changes.
        
        Output:
            MovementOutput: The steering force for wandering movement.
        """
        # Create a small random displacement
        random_displacement = np.random.uniform(-1, 1, size=2) * config.JITTER
        
        # Wander circle center (ahead of the player)
        wander_target = player.position + player.velocity.normalized() * config.WANDER_DIST
        
        # Apply random displacement to the circle
        wander_target += random_displacement * config.WANDER_RADIUS
        
        # Seek the new random target
        return SteeringBehaviors.seek(player, MovementInput(wander_target))

    @staticmethod
    def blend_steering_behaviors(actions: list, player: MovementInput, target: MovementInput,neighbors:list) -> MovementOutput:
        blended_output = MovementOutput(linear=np.array((0.0, 0.0)), angular=0.0)

        # Apply selected movements
        for behavior, weight in actions:
            output = behavior(player.object, target)
            blended_output.linear += np.dot(output.linear , weight)
            blended_output.angular += np.dot(output.angular , weight)

        # Always apply collision avoidance, separation, and obstacle avoidance
        #obstacle_output=SteeringBehaviors.obstacle_avoidance(player.object)
        separate_output=SteeringBehaviors.separation(player.object,neighbors)
        collision_output=SteeringBehaviors.collision_avoidance(player.object,target)

        avoidance_behaviors = [
            #obstacle_output,
            separate_output,
            collision_output
        ]
        INFLUENCE=0.5
        for behavior in avoidance_behaviors:
            blended_output.linear += np.dot(output.linear ,INFLUENCE)  # Moderate influence
            blended_output.angular += np.dot(output.angular ,INFLUENCE)

        # Limit to maximum speed and return the blended movement
        blended_output.linear = np.clip(blended_output.linear, -config.PLAYER_MAX_SPEED, config.PLAYER_MAX_SPEED)

        return blended_output

