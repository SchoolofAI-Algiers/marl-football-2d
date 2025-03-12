import numpy as np
import math
import random
import pygame
from env.utils import get_dimensions, get_physics, get_simulation
from env.models import GameState,MovementInput,MovementOutput
from env.engine import Object, Player, Ball, distance
from env.steering import SteeringBehaviors as steering
from env.config import CATCH_RADIUS,TACKLE_RADIUS
import env.config as config

class FootballEnv:
    def __init__(self, team_size=2):

        self.team_size = team_size
        self.num_players = 2 * team_size
        self.dimensions = get_dimensions(self.num_players)
        self.physics = get_physics()
        self.simulation = get_simulation()
        
        self._init_players()
        self.ball = Ball(
            object=Object(
                radius=self.dimensions.ball_radius,
                position=(self.dimensions.stadium_length / 2, self.dimensions.stadium_width / 2),
                velocity=(0, 0),
                rotation=0.0,
                angular_velocity=0.0
            )
        )
        
        self.game_state = GameState(
            game_time=0.0,
            score=[0, 0],
            possession=None,
            possession_time=[0.0, 0.0]
        )

       

        self.done = False

    def _init_players(self):
        """
        Positions players neatly on the field:
        - Team 0 (left side) at x ~ 20% of field_length
        - Team 1 (right side) at x ~ 80% of field_length
        - Spaced along the center on the y-axis
        """
        self.players = []
        for i in range(self.num_players):

            team = 0 if i < self.team_size else 1
            x = 0.2 * self.dimensions.stadium_length if team == 0 else 0.8 * self.dimensions.stadium_length
            
            # Distribute players around the center on y-axis
            idx_within_team = i % self.team_size
            center_y = self.dimensions.stadium_width / 2
            offset = (idx_within_team - (self.team_size - 1) / 2) * 5
            y = center_y + offset

            orientation = 0.0 if team == 0 else np.pi
            
            self.players.append(Player(
                object=Object(
                    radius=self.dimensions.player_radius,
                    position=(x, y),
                    velocity=(0, 0),
                    rotation=orientation,
                    angular_velocity=0.0
                ),
                team=team
            ))

    def _get_state(self):
        state = []            
        for player in self.players:
            state += [
                player.object.position[0], player.object.position[1],
                player.object.velocity[0], player.object.velocity[1],
                player.object.rotation, player.object.angular_velocity,
                player.team
            ]
        
        state += [
            self.ball.object.position[0], self.ball.object.position[1],
            self.ball.object.velocity[0], self.ball.object.velocity[1]
        ]
        
        poss_total = max(sum(self.game_state.possession_time), 1e-6)
        
        state += [
            self.game_state.score[0], self.game_state.score[1],
            self.game_state.possession if self.game_state.possession is not None else -1,
            self.game_state.possession_time[0] / poss_total,
            self.game_state.possession_time[1] / poss_total,
            self.game_state.game_time
        ]
        
        return np.array(state, dtype=np.float32)

    def reset(self, on_goal=False):
        """
        Resets the environment:
        - Re-initializes players
        - Resets ball position, score, possession, time
        """
        self._init_players()
        self.ball.object.position = (self.dimensions.stadium_length / 2, self.dimensions.stadium_width / 2)
        self.ball.object.velocity = (0, 0)
        
        if not on_goal:
            self.game_state.score = [0, 0]
            self.game_state.possession = None
            self.game_state.possession_time = [0.0, 0.0]
            self.game_state.game_time = 0.0
            self.done = False

        return self._get_state()
    
    def translate_action(self,object,action,neighbors,target=None):        
        if action=="pass":
            actions = [
                (steering.seek, 1.0),  # Seek towards the ball
                (steering.arrive, 0.5),  # Arrive at the ball with a slower speed
                
            ]
            if target==None:
                target=MovementInput(neighbors[0].position,neighbors[0].velocity,neighbors[0].rotation,neighbors[0].angular_velocity)

            rotation=steering.face(object.object,neighbors[0].object)

            ballOutput=steering.blend_steering_behaviors(actions,target,neighbors[0],neighbors)
            playerOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)

        elif action =='idle': #the player doesn't do anything
            playerOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)
            ballOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)
            rotation= object.object.rotation

        elif action =='follow': #the player goes after the ball
            actions = [
                (steering.seek, 1.0),  # Seek towards the ball
                (steering.arrive, 0.5),  # Arrive at the ball with a slower speed
                
            ]
            rotation=steering.face(object.object,target.object)
            playerOutput=steering.blend_steering_behaviors(actions,object,target,neighbors)
            ballOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)

        elif action=="catch": #the player attempts to catch the ball
            actions = [
                (steering.seek, 0.5),  # Seek towards the ball
                (steering.arrive, 1),  # Arrive at the ball with a slower speed  
            ]
            dist_to_ball = distance(object.object, self.ball.object)
            if dist_to_ball<= CATCH_RADIUS:
                playerOutput=steering.blend_steering_behaviors(actions,object,target,neighbors)

                ballOutput=steering.blend_steering_behaviors([actions[1]],object,target,neighbors)
                
            else:
                playerOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)
                ballOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)
            rotation= object.object.rotation

        elif action == "tackle":  # The player attempts to tackle the opponent with the ball
            actions = [
                (steering.pursue, 1.0),  # Predict opponent's movement and chase
                (steering.seek, 0.7),    # Directly chase if prediction fails
                (steering.arrive, 0.5),  # Slow down near opponent for a controlled tackle  
            ]
            
            dist_to_opponent = distance(object.object, target.object)  # Target is the opponent

            if dist_to_opponent <= TACKLE_RADIUS:
                playerOutput = steering.blend_steering_behaviors(actions, object, target, neighbors)
                ballOutput = MovementOutput(linear=[0.0, 0.0], angular=0.0)

                # Opponent slows down or stops if tackled successfully
                #opponentOutput = steering.blend_steering_behaviors([(steering.arrive, 1.0)], target, object, neighbors)
            else:
                playerOutput = MovementOutput(linear=[0.0, 0.0], angular=0.0)
                ballOutput = MovementOutput(linear=[0.0, 0.0], angular=0.0)

            rotation = object.object.rotation
        elif action=="wander":
            actions = [
                (steering.wander, 1.0),  
            ]
            playerOutput = steering.blend_steering_behaviors(actions, object, target, neighbors)
            ballOutput = MovementOutput(linear=[0.0, 0.0], angular=0.0)
            rotation = object.object.rotation

        elif action=="shoot":
            self.scale = 8
            

            def get_random_goal_target(goal_x, goal_y, goal_width, goal_height):
                """
                Returns a random point within the goal's boundaries.
                """
                target_x = goal_x + random.uniform(5, goal_width-5)  # Random x within goal width
                target_y = goal_y + random.uniform(5, goal_height-5)  # Random y within goal height
                return (target_x, target_y)

            # find the goals positions:
            field_length = self.dimensions.stadium_length
            field_width = self.dimensions.stadium_width
            # Goal dimensions 
            goal_width = self.dimensions.goal_area_width
            goal_height = self.dimensions.goal_area_length
         
            offset = (field_width -goal_width) // 2
            


            # Left goal boundaries
            left_goal_x = 0  # Left goal's x position
            left_goal_y = goal_height  # Left goal's top y position
            
            # Right goal boundaries
            right_goal_x = field_length - goal_height  # Right goal's x position
            right_goal_y = offset # Right goal's top y position
            
            if object.team==0:
                # Team 0 shoots at the right goal
                goal_x = right_goal_x
                goal_y = right_goal_y
            else:
                # Team 1 shoots at the left goal
                goal_x = left_goal_x
                goal_y = left_goal_y

            actions = [
                (steering.seek, 1.0),
                (steering.arrive, 0.5)  
            ]
            # the point chosen should be converted to array
            goal_point=get_random_goal_target(goal_x,goal_y,goal_width,goal_height)
            goal_point=np.array(goal_point)
            
            goal_target=MovementInput(position=goal_point,velocity=[0.0,0.0],rotation=0.0,angular_velocity=0.0)

            ballOutput=steering.blend_steering_behaviors(actions,target,goal_target,neighbors)
            playerOutput=MovementOutput(linear=[0.0, 0.0], angular=0.0)

            rotation=0.0

        return [playerOutput.linear[0], playerOutput.linear[1], playerOutput.angular ,ballOutput.linear[0],ballOutput.linear[1] ,rotation]




    def step(self, actions):

        """
        One simulation step:
        - Applies actions to players (acceleration, angular acceleration, kick)
        - Updates positions and ball dynamics
        - Checks for goals and updates score/rewards
        - Returns (state, rewards, done, info)
        """
        if self.done:
            return self._get_state(), [0] * self.num_players, True, {}
            
        rewards = [0.0] * self.num_players

        ball_acceleration = [0, 0]
        #possession = None
        sum_radii = (self.dimensions.player_radius 
                    + self.dimensions.ball_radius)
        
        # 1. First update ALL players' movements
        ball_rotation=self.ball.object.rotation
        for i, player in enumerate(self.players):
    


            ax, ay, alpha, kx, ky, rotation = self.translate_action(player,actions[i],self._get_neighbors(player,30),self.ball)

            dist_to_ball = distance(player.object, self.ball.object)
            
            # Use <= for contact detection
            if dist_to_ball <= sum_radii:
                ball_rotation=rotation
                ball_acceleration[0] += kx
                ball_acceleration[1] += ky
                self.game_state.possession= player.team
            
            player.object.act(
                acceleration=(ax, ay),
                angular_acceleration=alpha,
                dt=self.simulation.dt,
                max_speed=self.physics.player_max_speed,
                max_angular_speed=self.physics.player_max_rotation,
                friction_factor=self.physics.friction_factor,
                angular_friction_factor=self.physics.angular_friction_factor,
                min_length=0,
                max_length=self.dimensions.stadium_length,
                min_width=0,
                max_width=self.dimensions.stadium_width,
                target_rotation=rotation
                
            )
        
        
        #2. Then check for ball contacts AFTER all players moved
        for i, player in enumerate(self.players):
            dist_to_ball = distance(player.object, self.ball.object)
                
            # Use <= for contact detection
            if dist_to_ball <= sum_radii:  
                # Apply this player's kick force
                # kx, ky = actions[i][3], actions[i][4]
                ball_acceleration[0] += kx
                ball_acceleration[1] += ky
                self.game_state.possession= player.team  # Last contacting player gets possession

    

        if self.game_state.possession==0:
            self.ball.object.rotation=0.0
        else: 
            self.ball.object.rotation=np.pi

        # 3. Apply accumulated acceleration to ball
        self.ball.object.act(
            acceleration=ball_acceleration,
            angular_acceleration=0,
            dt=self.simulation.dt,
            max_speed=None,
            max_angular_speed=None,
            friction_factor=self.physics.air_resistance,
            angular_friction_factor=1,
            min_length=0,
            max_length=self.dimensions.stadium_length,
            min_width=0,
            max_width=self.dimensions.stadium_width,
            target_rotation=ball_rotation
        )
        
        # Update possession state
        #self.game_state.possession = possession
        if self.game_state.possession  is not None:
            self.game_state.possession_time[self.game_state.possession ] += self.simulation.dt
        
        # Check for goals
        if self.ball.object.position[0] < 0:
            self.game_state.score[1] += 1
            self.reset(on_goal=True)
            for i, player in enumerate(self.players):
                rewards[i] = 1.0 if player.team == 1 else -1.0

        elif self.ball.object.position[0] > self.dimensions.stadium_length:
            self.game_state.score[0] += 1
            self.reset(on_goal=True)
            for i, player in enumerate(self.players):
                rewards[i] = 1.0 if player.team == 0 else -1.0
        
        # Update game state
        self.game_state.game_time += self.simulation.dt
        self.done = self.game_state.game_time >= self.simulation.max_game_time
        
        return self._get_state(), rewards, self.done, {}
    

    def _get_neighbors(self,player,radius)->list:
        '''
        get the neighbors of the player within a certain radius
        '''
        neighbors = []
        player_position = np.array(player.object.position)

        for other_player in self.players:
            if other_player != player:  # Avoid self-comparison
                other_position = np.array(other_player.object.position)
                distance = np.linalg.norm(other_position - player_position)


                if distance <= radius:
                    neighbors.append(other_player)

        return neighbors

    def render(self):
        """
        Renders a realistic soccer pitch with:
        - Green background, white boundary lines, center line, center circle
        - Penalty boxes and goals drawn inside the boundaries
        - Circles representing players (red for team 0, blue for team 1)
        - A white circle for the ball
        """
        if not hasattr(self, 'screen'):
            # Initialize video system (if not already done)
            pygame.init()
            self.width_pixels = 800
            self.height_pixels = 480
            self.screen = pygame.display.set_mode((self.width_pixels, self.height_pixels))
            pygame.display.set_caption("2D Football Environment")
            self.scale = 8  # 1 game unit = 8 pixels

        # Fill background with a grass-green color
        self.screen.fill((34, 139, 34))

        # Convert field dimensions to pixel space
        field_px_length = self.dimensions.stadium_length * self.scale
        field_px_width = self.dimensions.stadium_width * self.scale

        # Center the field in the window
        offset_x = (self.width_pixels - field_px_length) // 2
        offset_y = (self.height_pixels - field_px_width) // 2

        # Draw outer boundary
        field_rect = pygame.Rect(offset_x, offset_y, field_px_length, field_px_width)
        pygame.draw.rect(self.screen, (255, 255, 255), field_rect, 2)

        # Draw center line
        center_x = offset_x + field_px_length // 2
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, offset_y), (center_x, offset_y + field_px_width), 2)

        # Draw center circle (radius ~10 game units)
        center_circle_radius = 10 * self.scale
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, offset_y + field_px_width // 2), center_circle_radius, 2)

        # Draw penalty boxes (simplified)
        box_width = 16.5 * self.scale
        box_height = 40.32 * self.scale
        top_box_y = offset_y + (field_px_width - box_height) / 2

        # Left penalty box
        left_box_rect = pygame.Rect(offset_x, top_box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (255, 255, 255), left_box_rect, 2)

        # Right penalty box
        right_box_rect = pygame.Rect(offset_x + field_px_length - box_width, top_box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (255, 255, 255), right_box_rect, 2)

        # Draw goals inside the boundaries for visibility
        goal_width = 2 * self.scale
        goal_height = 14 * self.scale
        top_goal_y = offset_y + (field_px_width - goal_height) / 2

        # Left goal
        left_goal_rect = pygame.Rect(offset_x, top_goal_y, goal_width, goal_height)
        pygame.draw.rect(self.screen, (255, 255, 255), left_goal_rect, 2)

        # Right goal
        right_goal_rect = pygame.Rect(offset_x + field_px_length - goal_width, top_goal_y, goal_width, goal_height)
        pygame.draw.rect(self.screen, (255, 255, 255), right_goal_rect, 2)

        # Draw players
        for player in self.players:
            px = offset_x + player.object.position[0] * self.scale
            py = offset_y + player.object.position[1] * self.scale
            color = (255, 0, 0) if player.team == 0 else (0, 0, 255)
            pygame.draw.circle(self.screen, color, (int(px), int(py)), int(self.dimensions.player_radius * self.scale))


            # Draw facing direction
            # facing_length = 10  # Standard length for the facing direction line
            # player_rotation = player.object.rotation  # Directly use the rotation attribute
            # end_x = px + facing_length * math.cos(player_rotation)
            # end_y = py - facing_length * math.sin(player_rotation)  # Subtract because y-axis is inverted in Pygame
            # pygame.draw.line(self.screen, (255, 255, 255), (px, py), (end_x, end_y), 2)
            
        # Draw ball
        bx = offset_x + self.ball.object.position[0] * self.scale
        by = offset_y + self.ball.object.position[1] * self.scale
        pygame.draw.circle(self.screen, (255, 165, 0), (int(bx), int(by)), int(self.dimensions.ball_radius * self.scale))

        
        pygame.display.flip()
