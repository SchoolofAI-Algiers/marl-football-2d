from decimal import Decimal
from typing import Tuple, List
import numpy as np
import pygame
from env.utils import get_dimensions, get_physics, get_simulation, random_float
from env.schema import GameState, PlayerState, TeamState, BallState, EnvironmentState, StepResult, PlayerReward, TeamRewards, PlayerAction, TeamActions
from env.engine import Object, Player, Ball, distance, resolve_collisions
from env.config import BLACK, WHITE, GREEN, ORANGE, YELLOW, RED, BLUE, RENDER_SCALE, PADDING, STEP_REWARD, KICK_REWARD

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
                orientation=0.0,
                angular_velocity=0.0,
                bounceable=True,
                restitution=1
            )
        )
        
        self.game_state = GameState(
            game_time=Decimal('0.0'),
            score=[0, 0],
            possession=None,
            possession_time=[0.0, 0.0],
            last_kicker=None,
            last_kick_time=None
        )

        self.done = False

    def _init_players(self):
        """
        Positions players neatly on the field:
        - Team 0 (left side) at x ~ 40% of field_length
        - Team 1 (right side) at x ~ 60% of field_length
        - Spaced along the center on the y-axis
        """
        self.players = []
        for i in range(self.num_players):
            team = 0 if i < self.team_size else 1
            x = 0.25 * self.dimensions.stadium_length if team == 0 else 0.75 * self.dimensions.stadium_length
            
            # Distribute players around the center on y-axis
            idx_within_team = i % self.team_size
            center_y = self.dimensions.stadium_width / 2
            offset = (idx_within_team - (self.team_size - 1) / 2) * 3
            y = center_y + offset

            orientation = 0.0 if team == 0 else np.pi
            
            self.players.append(Player(
                object=Object(
                    radius=self.dimensions.player_radius,
                    position=(x, y),
                    velocity=(0, 0),
                    orientation=orientation,
                    angular_velocity=0.0,
                    bounceable=True,
                    restitution=0.1
                ),
                team=team
            ))

    def _get_state(self, normalize: bool = True) -> EnvironmentState:
        def normalize_position(pos: Tuple[float, float]) -> Tuple[float, float]:
            return (
                pos[0] / self.dimensions.stadium_length,
                pos[1] / self.dimensions.stadium_width
            )

        def normalize_velocity(vel: Tuple[float, float], max_speed: float) -> Tuple[float, float]:
            return (
                vel[0] / max_speed,
                vel[1] / max_speed
            )

        def normalize_angle(angle: float) -> float:
            return angle / np.pi  # maps [-π, π] to [-1, 1]

        def normalize_angular_velocity(av: float) -> float:
            return av / self.physics.player_max_angular_speed

        def player_to_state(player) -> PlayerState:
            obj = player.object
            return PlayerState(
                position=normalize_position(obj.position) if normalize else obj.position,
                velocity=normalize_velocity(obj.velocity, self.physics.player_max_speed) if normalize else obj.velocity,
                orientation=normalize_angle(obj.orientation) if normalize else obj.orientation,
                angular_velocity=normalize_angular_velocity(obj.angular_velocity) if normalize else obj.angular_velocity
            )

        # Split players by team
        team1_players = [p for p in self.players if p.team == 0]
        team2_players = [p for p in self.players if p.team == 1]

        team1_state = TeamState(players=[player_to_state(p) for p in team1_players])
        team2_state = TeamState(players=[player_to_state(p) for p in team2_players])

        ball_obj = self.ball.object
        ball_state = BallState(
            position=normalize_position(ball_obj.position) if normalize else ball_obj.position,
            velocity=normalize_velocity(ball_obj.velocity, self.physics.ball_max_speed) if normalize else ball_obj.velocity
        )

        return EnvironmentState(
            team1=team1_state,
            team2=team2_state,
            ball=ball_state
        )

    def _denormalize_actions(self, actions: TeamActions) -> TeamActions:
        def denormalize_action(action: PlayerAction) -> PlayerAction:
            return PlayerAction(
                acceleration=action.acceleration * self.physics.player_max_acceleration,
                angular_acceleration=action.angular_acceleration * self.physics.player_max_angular_acceleration,
                kicking_force=action.kicking_force * self.physics.player_max_kicking_force,
                kicking_angle=action.kicking_angle * np.pi  # Convert to radians
            )
        return TeamActions(
            team1=[denormalize_action(a) for a in actions.team1],
            team2=[denormalize_action(a) for a in actions.team2]
        )

    def _handle_goal(self, scoring_team: int, team1_rewards: List[PlayerReward], team2_rewards: List[PlayerReward]) -> Tuple[List[PlayerReward], List[PlayerReward]]:
        # print(f"Goal scored by Team {scoring_team} at time {self.game_state.game_time:.2f}")
        self.game_state.score[scoring_team] += 1
        self.reset(on_goal=True)
        for i, player in enumerate(self.players):
            r = 1.0 if player.team == scoring_team else -1.0
            if player.team == 0:
                team1_rewards[i].reward += r
            else:
                team2_rewards[i - self.team_size].reward += r
        return team1_rewards, team2_rewards

    def reset(self, on_goal=False):
        """
        Resets the environment:
        - Re-initializes players
        - Resets ball position, score, possession, time
        """
        self._init_players()
        self.ball.object.position = (self.dimensions.stadium_length / 2, self.dimensions.stadium_width / 2)
        self.ball.object.velocity = (0, 0)
        
        if on_goal:
            self.game_state.possession = None
            self.game_state.last_kicker = None
            self.game_state.last_kick_time = None
        else:
            self.game_state = GameState(
                game_time=Decimal('0.0'),
                score=[0, 0],
                possession=None,
                possession_time=[Decimal('0.0'), Decimal('0.0')],
                last_kicker=None,
                last_kick_time=None
            )
            self.done = False

        return self._get_state()

    def step(self, actions: TeamActions):
        """
        One simulation step:
        - Applies actions to players (acceleration, angular acceleration, force power, force angle)
        - Updates positions and ball dynamics
        - Checks for goals and updates score/rewards
        - Returns (state, rewards, done, info)
        """
        if self.done:
            return self._get_state(), [0] * self.num_players, True, {}
        
        goal_min_y = (self.dimensions.stadium_width - self.dimensions.goal_width) / 2
        goal_max_y = goal_min_y + self.dimensions.goal_width

        denormalized_actions = self._denormalize_actions(actions)
        all_actions = denormalized_actions.team1 + denormalized_actions.team2
        
        team1_rewards, team2_rewards = [], []
        for i, player in enumerate(self.players):
            reward = PlayerReward(player_id=i, reward=STEP_REWARD)
            (team1_rewards if player.team == 0 else team2_rewards).append(reward)

        # 1. First update ALL players' movements
        for i, player in enumerate(self.players):
            action = all_actions[i]

            acceleration = action.acceleration
            angular_acceleration = action.angular_acceleration
            kicking_force = action.kicking_force
            kicking_angle = action.kicking_angle

            # Convert acceleration to x/y using player's orientation
            ax = np.cos(player.object.orientation) * acceleration
            ay = np.sin(player.object.orientation) * acceleration

            player.object.act(
                acceleration=(ax, ay),
                angular_acceleration=angular_acceleration,  
                dt=self.simulation.dt,
                max_speed=self.physics.player_max_speed,
                max_angular_speed=self.physics.player_max_angular_speed,
                friction_factor=self.physics.player_friction_factor,
                angular_friction_factor=self.physics.angular_friction_factor,
                min_length=0,
                max_length=self.dimensions.stadium_length,
                min_width=0,
                max_width=self.dimensions.stadium_width,
                goal_min_y=goal_min_y,
                goal_max_y=goal_max_y,
                goal_depth=self.dimensions.goal_depth
            )

        # Resolve collisions between players
        new_objects = resolve_collisions([player.object for player in self.players])
        for i, player in enumerate(self.players):
            player.object = new_objects[i]

        # 2. Then check for ball contacts AFTER all players moved
        ball_acceleration = np.array([0.0, 0.0])
        possession = None
        sum_radii = self.dimensions.player_radius + self.dimensions.ball_radius

        for i, player in enumerate(self.players):
            action = all_actions[i]
            dist_to_ball = distance(player.object, self.ball.object)

            # Allow kicks only if the player is not the last toucher OR if cooldown has passed
            if dist_to_ball <= sum_radii and (
                self.game_state.last_kicker == None or
                self.game_state.last_kicker != player or 
                (self.game_state.game_time - self.game_state.last_kick_time) > self.physics.kick_cooldown
            ):
                # Convert force angle to absolute angle
                kick_direction = player.object.orientation + action.kicking_angle

                # Convert force power to x/y using the force direction
                kick_x = np.cos(kick_direction) * action.kicking_force
                kick_y = np.sin(kick_direction) * action.kicking_force

                # print(f'KICK {player.team} {self.game_state.game_time:.2f} {action.kicking_force:.2f} ')

                # Apply the kick force to the ball
                ball_acceleration += np.array([kick_x, kick_y])
                
                # reward the player for kicking the ball
                if player.team == 0:
                    team1_rewards[i].reward += KICK_REWARD
                else:
                    team2_rewards[i - self.team_size].reward += KICK_REWARD

                # Update possession
                possession = player.team  

                # Track last kicker and time of kick
                self.game_state.last_kicker = player
                self.game_state.last_kick_time = self.game_state.game_time
        
        # 3. Apply accumulated acceleration to ball
        self.ball.object.act(
            acceleration=ball_acceleration,
            angular_acceleration=0,
            dt=self.simulation.dt,
            max_speed=self.physics.ball_max_speed,
            max_angular_speed=None,
            friction_factor=self.physics.ball_friction_factor,
            angular_friction_factor=1,
            min_length=0,
            max_length=self.dimensions.stadium_length,
            min_width=0,
            max_width=self.dimensions.stadium_width,
            goal_min_y=goal_min_y,
            goal_max_y=goal_max_y,
            goal_depth=self.dimensions.goal_depth,
            instant=True
        )

        # Update possession state
        self.game_state.possession = possession
        if possession is not None:
            self.game_state.possession_time[possession] += self.simulation.dt

        # Check for goal
        ball_x, ball_y = self.ball.object.position
        is_in_goal_y_range = goal_min_y <= ball_y <= goal_max_y

        if ball_x < 0 and is_in_goal_y_range:
            team1_rewards, team2_rewards = self._handle_goal(scoring_team=1, team1_rewards=team1_rewards, team2_rewards=team2_rewards)
        elif ball_x > self.dimensions.stadium_length and is_in_goal_y_range:
            team1_rewards, team2_rewards = self._handle_goal(scoring_team=0, team1_rewards=team1_rewards, team2_rewards=team2_rewards)

        # Update game state
        self.game_state.game_time += self.simulation.dt
        self.done = self.game_state.game_time >= self.simulation.max_game_time

        return StepResult(
            state=self._get_state(),
            rewards=TeamRewards(
                team1=team1_rewards,
                team2=team2_rewards
            ),
            done=self.done,
            info={}
        )

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
            self.width_pixels = (self.dimensions.stadium_length + 2*self.dimensions.goal_depth + PADDING) * self.scale
            self.height_pixels = (self.dimensions.stadium_width + 1.5 * PADDING) * self.scale
            self.screen = pygame.display.set_mode((self.width_pixels, self.height_pixels))
            pygame.display.set_caption("2D Football Environment")

        self.screen.fill(GREEN)
        
        def draw_rectangle(x, y, length, width, color=WHITE):
            pygame.draw.rect(self.screen, color, pygame.Rect(x, y, length, width), 2)
        
        def draw_circle(x, y, radius, color=WHITE):
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
            
        # Draw leaderboard background
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width_pixels, 0.375 * PADDING * self.scale))
        
        # Render leaderboard (score, time), centered
        font = pygame.font.SysFont("Consolas", int(0.25 * PADDING * self.scale))
        score_text = f"A {self.game_state.score[0]} - {self.game_state.score[1]} B  |  Time: {self.game_state.game_time:.1f} / {self.simulation.max_game_time}"
        text_surface = font.render(score_text, True, WHITE)
        self.screen.blit(text_surface, ((self.width_pixels - text_surface.get_width()) // 2, 0.375 / 2 * PADDING * self.scale - text_surface.get_height() // 2))

        # Draw field boundary 
        field_length_px = self.dimensions.stadium_length * self.scale
        field_width_px = self.dimensions.stadium_width * self.scale
        offset_x = (self.width_pixels - field_length_px) // 2
        offset_y = (self.height_pixels - field_width_px) // 2

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
        # goal_post_width_px = 2 * self.scale
        goal_depth_x = offset_x - self.dimensions.goal_depth * self.scale
        goal_depth_px = self.dimensions.goal_depth * self.scale
        goal_post_width_px = self.dimensions.goal_width * self.scale
        goal_post_top_y = offset_y + (field_width_px - goal_post_width_px) / 2
        
        draw_rectangle(goal_depth_x + 1, goal_post_top_y, goal_depth_px, goal_post_width_px, color=RED)
        draw_rectangle(offset_x + field_length_px - 1, goal_post_top_y, goal_depth_px, goal_post_width_px, color=BLUE)

        # Draw players
        for player in self.players:
            px = offset_x + player.object.position[0] * self.scale
            py = offset_y + player.object.position[1] * self.scale
            color = RED if player.team == 0 else BLUE
            draw_circle(px, py, self.dimensions.player_radius * self.scale, color)
            
            # Calculate the position of the small white dot
            dot_offset = self.dimensions.player_radius * self.scale * 0.5  # Move dot halfway to the edge
            angle = player.object.orientation  # Assuming orientation is in radians

            dot_x = px + np.cos(angle) * dot_offset
            dot_y = py + np.sin(angle) * dot_offset

            # Draw the white dot
            draw_circle(dot_x, dot_y, self.dimensions.player_radius * self.scale * 0.3, WHITE)

        # Draw ball
        ball_x = offset_x + self.ball.object.position[0] * self.scale
        ball_y = offset_y + self.ball.object.position[1] * self.scale
        draw_circle(ball_x, ball_y, self.dimensions.ball_radius * self.scale, YELLOW)
        
        pygame.display.flip()

class FootballNoOpponentEnv:
    def __init__(self, team_size=2):

        self.team_size = team_size
        self.num_players = team_size
        self.dimensions = get_dimensions(1.5 * self.num_players)
        self.physics = get_physics()
        self.simulation = get_simulation()
        
        self._init_players()
        self._init_ball(placement="random")
        
        self.game_state = GameState(
            game_time=Decimal('0.0'),
            score=[0, 0],
            possession=None,
            possession_time=[0.0, 0.0],
            last_kicker=None,
            last_kick_time=None
        )

        self.done = False

    def _init_players(self):
        self.players = []
        for i in range(self.num_players):
            team = 0
            x = np.random.random() * self.dimensions.stadium_length
            y = np.random.random() * self.dimensions.stadium_width

            orientation = 0.0 if team == 0 else np.pi
            
            self.players.append(Player(
                object=Object(
                    radius=self.dimensions.player_radius,
                    position=(x, y),
                    velocity=(0, 0),
                    orientation=orientation,
                    angular_velocity=0.0,
                    bounceable=True,
                    restitution=0.1
                ),
                team=team
            ))

    def _init_ball(self, placement: str = "center"):
        if placement == "center":
            x, y = self.dimensions.stadium_length / 2, self.dimensions.stadium_width / 2
        elif placement == "random":
            x, y = random_float() * self.dimensions.stadium_length, random_float() * self.dimensions.stadium_width
            vx, vy = random_float() * self.physics.ball_max_speed, random_float() * self.physics.ball_max_speed
            vm = 0.3
            vx, vy = vm * vx, vm * vy
        self.ball = Ball(
            object=Object(
                radius=self.dimensions.ball_radius,
                position=(x, y),
                velocity=(vx, vy),
                orientation=0.0,
                angular_velocity=0.0,
                bounceable=True,
                restitution=1
            )
        )

    def _get_state(self, normalize: bool = True) -> EnvironmentState:
        def normalize_position(pos: Tuple[float, float]) -> Tuple[float, float]:
            return (
                pos[0] / self.dimensions.stadium_length,
                pos[1] / self.dimensions.stadium_width
            )

        def normalize_velocity(vel: Tuple[float, float], max_speed: float) -> Tuple[float, float]:
            return (
                vel[0] / max_speed,
                vel[1] / max_speed
            )

        def normalize_angle(angle: float) -> float:
            return angle / np.pi  # maps [-π, π] to [-1, 1]

        def normalize_angular_velocity(av: float) -> float:
            return av / self.physics.player_max_angular_speed

        def player_to_state(player) -> PlayerState:
            obj = player.object
            return PlayerState(
                position=normalize_position(obj.position) if normalize else obj.position,
                velocity=normalize_velocity(obj.velocity, self.physics.player_max_speed) if normalize else obj.velocity,
                orientation=normalize_angle(obj.orientation) if normalize else obj.orientation,
                angular_velocity=normalize_angular_velocity(obj.angular_velocity) if normalize else obj.angular_velocity
            )

        # Split players by team
        team1_players = [p for p in self.players]

        team1_state = TeamState(players=[player_to_state(p) for p in team1_players])

        ball_obj = self.ball.object
        ball_state = BallState(
            position=normalize_position(ball_obj.position) if normalize else ball_obj.position,
            velocity=normalize_velocity(ball_obj.velocity, self.physics.ball_max_speed) if normalize else ball_obj.velocity
        )

        return EnvironmentState(
            team1=team1_state,
            ball=ball_state
        )

    def _denormalize_actions(self, actions: TeamActions) -> TeamActions:
        def denormalize_action(action: PlayerAction) -> PlayerAction:
            return PlayerAction(
                acceleration=action.acceleration * self.physics.player_max_acceleration,
                angular_acceleration=action.angular_acceleration * self.physics.player_max_angular_acceleration,
                kicking_force=action.kicking_force * self.physics.player_max_kicking_force,
                kicking_angle=action.kicking_angle * np.pi  # Convert to radians
            )
        return TeamActions(
            team1=[denormalize_action(a) for a in actions.team1]
        )

    def _handle_goal(self, scoring_team: int, team1_rewards: List[PlayerReward]) -> List[PlayerReward]:
        # print(f"Goal scored by Team {scoring_team} at time {self.game_state.game_time:.2f}")
        self.game_state.score[scoring_team] += 1
        self.reset(on_goal=True)
        for i, player in enumerate(self.players):
            r = 1.0 if player.team == scoring_team else -1.0
            team1_rewards[i].reward += r
        return team1_rewards

    def reset(self, on_goal=False):
        """
        Resets the environment:
        - Re-initializes players
        - Resets ball position, score, possession, time
        """
        self._init_players()
        self._init_ball(placement="random")
        
        if on_goal:
            self.game_state.possession = None
            self.game_state.last_kicker = None
            self.game_state.last_kick_time = None
        else:
            self.game_state = GameState(
                game_time=Decimal('0.1'),
                score=[0, 0],
                possession=None,
                possession_time=[Decimal('0.0'), Decimal('0.0')],
                last_kicker=None,
                last_kick_time=None
            )
            self.done = False

        return self._get_state()

    def step(self, actions: TeamActions):
        """
        One simulation step:
        - Applies actions to players (acceleration, angular acceleration, force power, force angle)
        - Updates positions and ball dynamics
        - Checks for goals and updates score/rewards
        - Returns (state, rewards, done, info)
        """
        if self.done:
            return StepResult(
                state=self._get_state(),
                rewards=TeamRewards(
                    team1=[PlayerReward(player_id=i, reward=0) for i in range(self.num_players)],
                    team2=None
                ),
                done=True,
                info={}
            )
        
        goal_min_y = (self.dimensions.stadium_width - self.dimensions.goal_width) / 2
        goal_max_y = goal_min_y + self.dimensions.goal_width

        denormalized_actions = self._denormalize_actions(actions)
        all_actions = denormalized_actions.team1
        
        team1_rewards = []
        for i, player in enumerate(self.players):
            reward = PlayerReward(player_id=i, reward=STEP_REWARD)
            team1_rewards.append(reward)

        # 1. First update ALL players' movements
        for i, player in enumerate(self.players):
            action = all_actions[i]

            acceleration = action.acceleration
            angular_acceleration = action.angular_acceleration
            kicking_force = action.kicking_force
            kicking_angle = action.kicking_angle

            # Convert acceleration to x/y using player's orientation
            ax = np.cos(player.object.orientation) * acceleration
            ay = np.sin(player.object.orientation) * acceleration

            player.object.act(
                acceleration=(ax, ay),
                angular_acceleration=angular_acceleration,  
                dt=self.simulation.dt,
                max_speed=self.physics.player_max_speed,
                max_angular_speed=self.physics.player_max_angular_speed,
                friction_factor=self.physics.player_friction_factor,
                angular_friction_factor=self.physics.angular_friction_factor,
                min_length=0,
                max_length=self.dimensions.stadium_length,
                min_width=0,
                max_width=self.dimensions.stadium_width,
                goal_min_y=goal_min_y,
                goal_max_y=goal_max_y,
                goal_depth=self.dimensions.goal_depth
            )

        # Resolve collisions between players
        new_objects = resolve_collisions([player.object for player in self.players])
        for i, player in enumerate(self.players):
            player.object = new_objects[i]

        # 2. Then check for ball contacts AFTER all players moved
        ball_acceleration = np.array([0.0, 0.0])
        possession = None
        sum_radii = self.dimensions.player_radius + self.dimensions.ball_radius

        for i, player in enumerate(self.players):
            action = all_actions[i]
            dist_to_ball = distance(player.object, self.ball.object)

            # Allow kicks only if the player is not the last toucher OR if cooldown has passed
            if dist_to_ball <= sum_radii and (
                self.game_state.last_kicker == None or
                self.game_state.last_kicker != player or 
                (self.game_state.game_time - self.game_state.last_kick_time) > self.physics.kick_cooldown
            ):
                # Convert force angle to absolute angle
                kick_direction = player.object.orientation + action.kicking_angle

                # Convert force power to x/y using the force direction
                kick_x = np.cos(kick_direction) * action.kicking_force
                kick_y = np.sin(kick_direction) * action.kicking_force

                # print(f'KICK {player.team} {self.game_state.game_time:.2f} {action.kicking_force:.2f} ')

                # Apply the kick force to the ball
                ball_acceleration += np.array([kick_x, kick_y])
                
                # reward the player for kicking the ball
                team1_rewards[i].reward += KICK_REWARD

                # Update possession
                possession = player.team  

                # Track last kicker and time of kick
                self.game_state.last_kicker = player
                self.game_state.last_kick_time = self.game_state.game_time
        
        # 3. Apply accumulated acceleration to ball
        self.ball.object.act(
            acceleration=ball_acceleration,
            angular_acceleration=0,
            dt=self.simulation.dt,
            max_speed=self.physics.ball_max_speed,
            max_angular_speed=None,
            friction_factor=self.physics.ball_friction_factor,
            angular_friction_factor=1,
            min_length=0,
            max_length=self.dimensions.stadium_length,
            min_width=0,
            max_width=self.dimensions.stadium_width,
            goal_min_y=goal_min_y,
            goal_max_y=goal_max_y,
            goal_depth=self.dimensions.goal_depth,
            instant=True
        )

        # Update possession state
        self.game_state.possession = possession
        if possession is not None:
            self.game_state.possession_time[possession] += self.simulation.dt

        # Check for goal
        ball_x, ball_y = self.ball.object.position
        is_in_goal_y_range = goal_min_y <= ball_y <= goal_max_y

        if ball_x < 0 and is_in_goal_y_range:
            team1_rewards = self._handle_goal(scoring_team=1, team1_rewards=team1_rewards)
        elif ball_x > self.dimensions.stadium_length and is_in_goal_y_range:
            team1_rewards = self._handle_goal(scoring_team=0, team1_rewards=team1_rewards)

        # Update game state
        self.game_state.game_time += self.simulation.dt
        self.done = self.game_state.game_time >= self.simulation.max_game_time

        return StepResult(
            state=self._get_state(),
            rewards=TeamRewards(
                team1=team1_rewards,
                team2=None
            ),
            done=self.done,
            info={}
        )

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
            self.width_pixels = (self.dimensions.stadium_length + 2*self.dimensions.goal_depth + PADDING) * self.scale
            self.height_pixels = (self.dimensions.stadium_width + 1.5 * PADDING) * self.scale
            self.screen = pygame.display.set_mode((self.width_pixels, self.height_pixels))
            pygame.display.set_caption("2D Football Environment")

        self.screen.fill(GREEN)
        
        def draw_rectangle(x, y, length, width, color=WHITE):
            pygame.draw.rect(self.screen, color, pygame.Rect(x, y, length, width), 2)
        
        def draw_circle(x, y, radius, color=WHITE):
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
            
        # Draw leaderboard background
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width_pixels, 0.375 * PADDING * self.scale))
        
        # Render leaderboard (score, time), centered
        font = pygame.font.SysFont("Consolas", int(0.25 * PADDING * self.scale))
        score_text = f"A {self.game_state.score[0]} - {self.game_state.score[1]} B  |  Time: {self.game_state.game_time:.1f} / {self.simulation.max_game_time}"
        text_surface = font.render(score_text, True, WHITE)
        self.screen.blit(text_surface, ((self.width_pixels - text_surface.get_width()) // 2, 0.375 / 2 * PADDING * self.scale - text_surface.get_height() // 2))

        # Draw field boundary 
        field_length_px = self.dimensions.stadium_length * self.scale
        field_width_px = self.dimensions.stadium_width * self.scale
        offset_x = (self.width_pixels - field_length_px) // 2
        offset_y = (self.height_pixels - field_width_px) // 2

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
        # goal_post_width_px = 2 * self.scale
        goal_depth_x = offset_x - self.dimensions.goal_depth * self.scale
        goal_depth_px = self.dimensions.goal_depth * self.scale
        goal_post_width_px = self.dimensions.goal_width * self.scale
        goal_post_top_y = offset_y + (field_width_px - goal_post_width_px) / 2
        
        draw_rectangle(goal_depth_x + 1, goal_post_top_y, goal_depth_px, goal_post_width_px, color=RED)
        draw_rectangle(offset_x + field_length_px - 1, goal_post_top_y, goal_depth_px, goal_post_width_px, color=BLUE)

        # Draw players
        for player in self.players:
            px = offset_x + player.object.position[0] * self.scale
            py = offset_y + player.object.position[1] * self.scale
            color = RED if player.team == 0 else BLUE
            draw_circle(px, py, self.dimensions.player_radius * self.scale, color)
            
            # Calculate the position of the small white dot
            dot_offset = self.dimensions.player_radius * self.scale * 0.5  # Move dot halfway to the edge
            angle = player.object.orientation  # Assuming orientation is in radians

            dot_x = px + np.cos(angle) * dot_offset
            dot_y = py + np.sin(angle) * dot_offset

            # Draw the white dot
            draw_circle(dot_x, dot_y, self.dimensions.player_radius * self.scale * 0.3, WHITE)

        # Draw ball
        ball_x = offset_x + self.ball.object.position[0] * self.scale
        ball_y = offset_y + self.ball.object.position[1] * self.scale
        draw_circle(ball_x, ball_y, self.dimensions.ball_radius * self.scale, YELLOW)
        
        pygame.display.flip()
