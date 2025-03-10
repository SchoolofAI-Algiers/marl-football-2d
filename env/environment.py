import numpy as np
import pygame
from env.utils import get_dimensions, get_physics, get_simulation
from env.models import GameState
from env.engine import Object, Player, Ball, distance
from config import WHITE, GREEN, ORANGE, RENDER_SCALE, PADDING

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
        
        # 1. First update ALL players' movements
        for i, player in enumerate(self.players):
            ax, ay, alpha, kx, ky = actions[i]
            
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
                max_width=self.dimensions.stadium_width
            )
        
        # 2. Then check for ball contacts AFTER all players moved
        ball_acceleration = [0.0, 0.0]
        possession = None
        sum_radii = (self.dimensions.player_radius 
                    + self.dimensions.ball_radius)
        
        for i, player in enumerate(self.players):
            dist_to_ball = distance(player.object, self.ball.object)
            
            # Use <= for contact detection
            if dist_to_ball <= sum_radii:  
                # Apply this player's kick force
                kx, ky = actions[i][3], actions[i][4]
                ball_acceleration[0] += kx
                ball_acceleration[1] += ky
                possession = player.team  # Last contacting player gets possession
        
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
            max_width=self.dimensions.stadium_width
        )
        
        # Update possession state
        self.game_state.possession = possession
        if possession is not None:
            self.game_state.possession_time[possession] += self.simulation.dt
        
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

    def render(self):
        """
        Renders a realistic soccer pitch with:
        - Green background, white boundary lines, center line, center circle
        - Penalty boxes and goals drawn inside the boundaries
        - Circles representing players (red for team 0, blue for team 1)
        - A white circle for the ball
        """
        
        if not hasattr(self, 'screen'):
            pygame.init()
            self.scale = RENDER_SCALE
            self.width_pixels = (self.dimensions.stadium_length + PADDING) * self.scale
            self.height_pixels = (self.dimensions.stadium_width + PADDING) * self.scale
            self.screen = pygame.display.set_mode((self.width_pixels, self.height_pixels))
            pygame.display.set_caption("2D Football Environment")

        self.screen.fill(GREEN)
        
        def draw_rectangle(x, y, length, width):
            pygame.draw.rect(self.screen, WHITE, pygame.Rect(x, y, length, width), 2)
        
        def draw_circle(x, y, radius, color=WHITE):
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))

        field_length_px = self.dimensions.stadium_length * self.scale
        field_width_px = self.dimensions.stadium_width * self.scale
        offset_x = (self.width_pixels - field_length_px) // 2
        offset_y = (self.height_pixels - field_width_px) // 2

        # Draw field boundary
        draw_rectangle(offset_x, offset_y, field_length_px, field_width_px)
        
        # Draw center line and circle
        center_x = offset_x + field_length_px // 2
        center_y = offset_y + field_width_px // 2
        pygame.draw.line(self.screen, WHITE, (center_x, offset_y), (center_x, offset_y + field_width_px), 2)
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), self.dimensions.center_circle_radius * self.scale, 2)

        # Draw penalty areas
        box_length_px = self.dimensions.penalty_area_length * self.scale
        box_width_px = self.dimensions.penalty_area_width * self.scale
        goal_area_length_px = self.dimensions.goal_area_length * self.scale
        goal_area_width_px = self.dimensions.goal_area_width * self.scale
        
        penalty_top_y = offset_y + (field_width_px - box_width_px) / 2
        goal_top_y = offset_y + (field_width_px - goal_area_width_px) / 2

        draw_rectangle(offset_x, penalty_top_y, box_length_px, box_width_px)
        draw_rectangle(offset_x + field_length_px - box_length_px, penalty_top_y, box_length_px, box_width_px)
        draw_rectangle(offset_x, goal_top_y, goal_area_length_px, goal_area_width_px)
        draw_rectangle(offset_x + field_length_px - goal_area_length_px, goal_top_y, goal_area_length_px, goal_area_width_px)

        # Draw penalty spots
        
        penalty_spot_radius_px = self.dimensions.penalty_spot_radius * self.scale
        left_penalty_x = offset_x + self.dimensions.penalty_spot_distance * self.scale
        right_penalty_x = offset_x + field_length_px - self.dimensions.penalty_spot_distance * self.scale
        
        draw_circle(left_penalty_x, center_y, penalty_spot_radius_px)
        draw_circle(right_penalty_x, center_y, penalty_spot_radius_px)

        # Draw goals
        goal_post_width_px = 2 * self.scale
        goal_post_height_px = self.dimensions.goal_width * self.scale
        goal_post_top_y = offset_y + (field_width_px - goal_post_height_px) / 2
        
        draw_rectangle(offset_x / 2 + 1, goal_post_top_y, offset_x / 2, goal_post_height_px)
        draw_rectangle(offset_x + field_length_px - 1, goal_post_top_y, offset_x / 2, goal_post_height_px)

        # Draw players
        for player in self.players:
            px = offset_x + player.object.position[0] * self.scale
            py = offset_y + player.object.position[1] * self.scale
            color = (255, 0, 0) if player.team == 0 else (0, 0, 255)
            draw_circle(px, py, self.dimensions.player_radius * self.scale, color)

        # Draw ball
        ball_x = offset_x + self.ball.object.position[0] * self.scale
        ball_y = offset_y + self.ball.object.position[1] * self.scale
        draw_circle(ball_x, ball_y, self.dimensions.ball_radius * self.scale, ORANGE)
        
        pygame.display.flip()
