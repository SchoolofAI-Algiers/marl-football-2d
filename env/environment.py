import numpy as np
import pygame
from env.utils import get_dimensions, get_physics, get_simulation
from env.models import GameState,MovementInput
from env.engine import Object, Player, Ball, distance

from env.steering import SteeringBehaviors
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
        #added this as a suggested modification to the environment
        self.env_state=[]

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
        print('getting state')
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
        print('done getting state')
        
        return np.array(state, dtype=np.float32)
    

# 
    def _get_state_dict(self):
        print("Getting state...")

        state = {
            "team_0": [],
            "team_1": [],
            "ball": {},
            "stadium": {}
        }

        # Organize player data by team
        for player in self.players:
            player_data = {
                "position": list(player.object.position),
                "velocity": list(player.object.velocity),
                "rotation": player.object.rotation,
                "angular_velocity": player.object.angular_velocity,
            }
            state[f"team_{player.team}"].append(player_data)

        # Ball state
        state["ball"] = {
            "position": list(self.ball.object.position),
            "velocity": list(self.ball.object.velocity)
        }

        # Normalize possession time
        poss_total = max(sum(self.game_state.possession_time), 1e-6)

        # Stadium (game state) information
        state["stadium"] = {
            "score": list(self.game_state.score),
            "possession": self.game_state.possession if self.game_state.possession is not None else -1,
            "possession_time": [
                self.game_state.possession_time[0] / poss_total,
                self.game_state.possession_time[1] / poss_total
            ],
            "game_time": self.game_state.game_time
        }

        print("Done getting state.")
        return state


    def reset(self, on_goal=False):
        """
        Resets the environment:
        - Re-initializes players
        - Resets ball position, score, possession, time
        """
        self._init_players()
        self.ball.object.position = (self.dimensions.stadium_length / 2, self.dimensions.stadium_width / 2)
        self.ball.object.velocity = (0, 0)
        print('game is reset')
        
        if not on_goal:
            self.game_state.score = [0, 0]
            self.game_state.possession = None
            self.game_state.possession_time = [0.0, 0.0]
            self.game_state.game_time = 0.0
            self.done = False

        return self._get_state()

    def step(self):
        """
        One simulation step:
        - Applies actions to players (acceleration, angular acceleration, kick)
        - Updates positions and ball dynamics
        - Checks for goals and updates score/rewards
        - Returns (state, rewards, done, info)
        """
        if self.done:
            return self._get_state_dict(), [0] * self.num_players, True, {}
            
        rewards = [0.0] * self.num_players
        
        # Update each player
        for i, player in enumerate(self.players):
            # ax, ay, alpha, kx, ky = actions[i]
            
            
            # player.object.act(
            #     acceleration=(ax, ay),
            #     angular_acceleration=alpha,
            #     dt=self.simulation.dt,
            #     max_speed=self.physics.player_max_speed,
            #     max_angular_speed=self.physics.player_max_rotation,
            #     friction_factor=self.physics.friction_factor,
            #     angular_friction_factor=self.physics.angular_friction_factor,
            #     min_length=0,
            #     max_length=self.dimensions.stadium_length,
            #     min_width=0,
            #     max_width=self.dimensions.stadium_width
            # )

             # Get the current state of the player
            player_input = MovementInput(
                position=player.object.position,
                velocity=player.object.velocity,
                rotation=player.object.rotation,
                angular_velocity=player.object.angular_velocity
            )
        
            # Call takeAction to determine the desired movement
            movement_output = self.takeAction(player, self._get_state_dict())
            
            # Apply the movement output to the player
            player.object.act(
                acceleration=movement_output.linear,
                angular_acceleration=movement_output.angular,
                dt=self.simulation.dt,
                max_speed=self.physics.player_max_speed,
                max_angular_speed=self.physics.player_max_rotation,
                friction_factor=self.physics.friction_factor,
                angular_friction_factor=self.physics.angular_friction_factor,
                min_length=0,
                max_length=self.dimensions.stadium_length,
                min_width=0,
                max_width=self.dimensions.stadium_width
            )
        
            dist_to_ball = distance(player.object, self.ball.object)
            if dist_to_ball < self.dimensions.player_radius + self.dimensions.ball_radius:
                ball_acceleration = [0.0, 0.0] #made it into 0, 0 just for testing , have not implemented the ball logic yet
                self.game_state.possession = player.team
            else:
                ball_acceleration = [0, 0]

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
            max_width=self.dimensions.stadium_width
        )
        
        if self.ball.object.position[0] < 0:
            self.game_state.score[1] += 1
            self.reset(on_goal=True)
            for i, player in enumerate(self.players):
                rewards[i] = 1.0 if player.team == 1 else -1.0

        elif self.ball.object.position[0] > self.dimensions.stadium_length:
            self.game_state.score[0] += 1
            self.reset(on_goal=True)
            for i, player in enumerate(self.players):
                rewards[i] = 1.0 if player['team'] == 0 else -1.0
        
        # Update game state
        self.game_state.game_time += self.simulation.dt
        if self.game_state.possession is not None:
            self.game_state.possession_time[self.game_state.possession] += self.simulation.dt
        self.done = self.game_state.game_time >= self.simulation.max_game_time
        
        return self._get_state(), rewards, self.done, {}

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

        # Draw ball
        bx = offset_x + self.ball.object.position[0] * self.scale
        by = offset_y + self.ball.object.position[1] * self.scale
        pygame.draw.circle(self.screen, (255, 255, 255), (int(bx), int(by)), int(self.dimensions.ball_radius * self.scale))

        pygame.display.flip()

    # function that makes the action using the steering algorithms:
    def takeAction(self,player_input,state):
        '''
        input: 
        - environment current state
        - optional: transition marix in case of high level env
        - player_input

        process: 
        - calls the select movements -> get the right actions
        - calls the apply movement -> blend algorithms based on the select movement output
        
        output:
        - movementOutput: the desired acceleration , position

        '''

        #Call selectAction to get the target position and actions with weights
        target_position, actions = self.selectAction(state, player_input)
        
        # Step 2: Blend the selected actions using the blending algorithm
        blended_output = SteeringBehaviors.blend_steering_behaviors(actions, player_input, target_position,self._get_neighbors(player_input,4))
        
        return blended_output
    
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

    #function used to define the best set of actions of the agent
    # def selectAction(state, player_input):
    #     '''
    #     input:
    #     - transition matrix (will be define as a self attribute in the high level environment)
    #     - environment state
    #     - player input

    #     process:
    #     - depends on the environment

    #     output:
    #     - Traget information (position)
    #     - Actions [] and their weights 
    #     '''
    
        


# suggested approach number 2:
'''
creating 2 subclasses:
- FootballLowEnv:
    inherits everything from the original environment, overrides the takeAction and the select-action function
    (the selection is done using an AI algorithm)

- FootballHighEnv:
    - inherits everything from the original environment, overrides the takeAction and the select-action function
    (the selection is done using a predefined transition matrix)
    - renderig might be overridden too (to consider the controller of the actions - mouse/keyboardw)


'''

# The low level environment
class FootballLowEnv(FootballEnv):

    def selectAction(self, state: np.array, player_input: MovementInput) -> tuple:
        """
        Selects the best set of actions for the agent using AI-based decision-making.
        
        Input:
            state (np.array): The current state of the environment.
            player_input (MovementInput): The current state of the player.
        
        Output:
            tuple: A tuple containing the target position and a list of actions with their weights.
        """
        

        #here goes the mastermind code to decide the behaviors intelligently
        
        # Example decision of the needed actions 
        actions = [
            (SteeringBehaviors.seek, 1.0),  # Seek towards the ball
            (SteeringBehaviors.arrive, 0.5)  # Arrive at the ball with a slower speed
        ]
        # the target position is decided according the action chosen: in this case we want to go after the ball so the position of the ball is taken as target
        target_position = state[-4:-2]  # Ball's position (assuming it's stored in the last four elements of the state)
        
        print(f"target position from selection{target_position}")
        return target_position, actions
    
# The high level environment
class FootballHighEnv(FootballEnv):
    def __init__(self, team_size=2):
        super().__init__(team_size)
        # a simplified example of the transition matrix
        self.transition_matrix = {
            'has_ball': {
                'pass': 0.6,  # 60% chance to pass
                'shoot': 0.3,  # 30% chance to shoot
                'dribble': 0.1  # 10% chance to dribble
            },
            'no_ball': {
                'seek': 0.8,  # 80% chance to seek the ball
                'support': 0.2  # 20% chance to support a teammate
            }
        }
        


    
    def selectAction(self, state: np.array, player_input: Player) -> tuple:
        """
        Selects the best set of actions for the agent using a predefined transition matrix.
        
        Input:
            state (np.array): The current state of the environment.
            player_input (MovementInput): The current state of the player.
        
        Output:
            tuple: A tuple containing the target position and a list of actions with their weights.
        """
        # Determine if the player has the ball
        print(f"input player: {player_input}")
        print(f"ball: {state['ball']}")
        distance= np.linalg.norm(np.array(player_input.object.position)-np.array(state['ball']['position']))
        
        has_ball = distance <= (self.dimensions.player_radius + self.dimensions.ball_radius)
        print(f'has ball {has_ball}')
        
        #Uses the transition matrix
        if has_ball:
            action = np.random.choice(list(self.transition_matrix['has_ball'].keys()), p=list(self.transition_matrix['has_ball'].values()))
        else:
            action = np.random.choice(list(self.transition_matrix['no_ball'].keys()), p=list(self.transition_matrix['no_ball'].values()))
        
        # Define the target position and actions based on the selected action
        if action == 'seek':
            target_position = MovementInput(position=state['ball']['position'], velocity=state['ball']['velocity'],rotation=0,angular_velocity=0) # Ball's position
            actions = [(SteeringBehaviors.seek, 1.0),(SteeringBehaviors.arrive, 0.3)]
        elif action == 'pass':
            target_position = self._get_teammate_position(player_input.team)  # Teammate's position
            actions = [(SteeringBehaviors.seek, 1.0)]
        elif action == 'shoot':
            target_position = MovementInput(position=[self.dimensions.stadium_length, self.dimensions.stadium_width / 2], velocity=[0.0,0.0],rotation=0,angular_velocity=0)  #goal position
            actions = [(SteeringBehaviors.seek, 1.0)]
        elif action == 'dribble':
            target_position = MovementInput(position=state['ball']['position'], velocity=state['ball']['velocity'],rotation=0,angular_velocity=0)   # Ball's position
            actions = [(SteeringBehaviors.arrive, 0.3),(SteeringBehaviors.seek, 0.7)]
        elif action == 'support':
            target_position = self._get_oppenent_position(player_input.team)  # Support position
            actions = [(SteeringBehaviors.pursue, 0.9),(SteeringBehaviors.arrive, 0.3)]
        
        return target_position, actions
    

    def _get_teammate_position(self, team: int) -> np.array:
        """
        Returns the position of a teammate.
        """
        
        for player in self.players:
            if player.team == team :
                return player.object
        return np.array([0, 0])
    
    def _get_oppenent_position(self, team: int) -> np.array:
        """
        Returns a support position based on the team's strategy.
        """
        for player in self.players:
            if player.team != team :
                return player.object
        return np.array([0, 0])
    


    
   