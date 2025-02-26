from typing import List

import numpy as np
import pygame
from env.utils import get_dimensions, get_physics, get_simulation
from env.models import GameState
from env.engine import Object, Player, Ball, distance

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
        
        # Update each player
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
        
            dist_to_ball = distance(player.object, self.ball.object)
            if dist_to_ball < self.dimensions.player_radius + self.dimensions.ball_radius:
                ball_acceleration = [kx, ky]
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
