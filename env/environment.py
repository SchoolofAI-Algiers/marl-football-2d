from typing import Tuple, List, Dict, Optional, Callable, Union
import functools

import numpy as np
import pygame
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

from env.utils import (
    get_dimensions, get_physics, get_simulation, random_float,
    mirror_position, mirror_velocity, mirror_action,
)
from env.schema import GameState
from env.engine import Object, Player, Ball, distance, resolve_collisions
from env.config import (
    BLACK, WHITE, GREEN, YELLOW, RED, BLUE,
    RENDER_SCALE, PADDING, STEP_REWARD, KICK_REWARD,
    PROXIMITY_REWARD_SCALE, BALL_POSITION_REWARD_SCALE,
    CORNER_AVOIDANCE_REWARD_SCALE,
)


class FootballEnv(ParallelEnv):
    """Unified 2D football environment implementing the PettingZoo Parallel API.

    Args:
        team_size: Number of players per team.
        opponent: Controls the opponent team configuration.
            - None: no opponent players on the field (single-team training).
            - "agents": both teams are PettingZoo agents (full MARL / self-play).
            - callable: opponent controlled by the given policy function;
              only team0 players are PettingZoo agents. Callable signature:
              (observations: dict[str, ndarray]) -> dict[str, ndarray].
              Observations are mirrored so the opponent sees the field as team0.
        render_mode: "human", "rgb_array", or None.
        ball_placement: "center" or "random".
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
        "name": "football_2d_v0",
    }

    def __init__(
        self,
        team_size: int = 1,
        opponent: Union[None, str, Callable] = "agents",
        render_mode: Optional[str] = None,
        ball_placement: str = "center",
        max_game_time: Optional[float] = None,
    ):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert ball_placement in ("center", "random")

        self.team_size = team_size
        self.ball_placement = ball_placement
        self.render_mode = render_mode

        if opponent is None:
            self._opponent_mode = "none"
            self._opponent_policy = None
        elif opponent == "agents":
            self._opponent_mode = "agents"
            self._opponent_policy = None
        elif callable(opponent):
            self._opponent_mode = "policy"
            self._opponent_policy = opponent
        else:
            raise ValueError(f"opponent must be None, 'agents', or a callable, got: {opponent!r}")

        self._has_opponent = self._opponent_mode != "none"
        self.num_players = team_size * (2 if self._has_opponent else 1)

        dim_players = self.num_players if self._has_opponent else 1.5 * self.num_players
        self.dimensions = get_dimensions(dim_players)
        self.physics = get_physics(dim_players)
        self.simulation = get_simulation()
        if max_game_time is not None:
            self.simulation.max_game_time = max_game_time

        team0_agents = [f"team0_player{i}" for i in range(team_size)]
        team1_agents = [f"team1_player{i}" for i in range(team_size)] if self._opponent_mode == "agents" else []
        self.possible_agents = team0_agents + team1_agents

        self._opponent_virtual_agents = (
            [f"team1_player{i}" for i in range(team_size)]
            if self._opponent_mode == "policy" else []
        )

        opp_size = team_size if self._has_opponent else 0
        self._per_player_obs_dim = 7 + (team_size - 1) * 7 + opp_size * 7 + 4 + 3 + 1 + 1

        self._screen = None
        self._clock = None

        self.players: List[Player] = []
        self.ball: Optional[Ball] = None
        self.game_state: Optional[GameState] = None
        self._done = False
        self._reward_components: List[Dict[str, float]] = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(self._per_player_obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

    def _init_players(self):
        self.players = []
        if self._has_opponent:
            for i in range(self.num_players):
                team = 0 if i < self.team_size else 1
                x = 0.25 * self.dimensions.stadium_length if team == 0 else 0.75 * self.dimensions.stadium_length

                idx_within_team = i % self.team_size
                center_y = self.dimensions.stadium_width / 2
                offset = (idx_within_team - (self.team_size - 1) / 2) * 3
                y = center_y + offset

                orientation = 0.0 if team == 0 else np.pi

                self.players.append(Player(
                    object=Object(
                        radius=self.dimensions.player_radius,
                        position=(x, y), velocity=(0, 0),
                        orientation=orientation, angular_velocity=0.0,
                        bounceable=True, restitution=0.1,
                    ),
                    team=team,
                ))
        else:
            for i in range(self.num_players):
                x = np.random.random() * self.dimensions.stadium_length
                y = np.random.random() * self.dimensions.stadium_width

                self.players.append(Player(
                    object=Object(
                        radius=self.dimensions.player_radius,
                        position=(x, y), velocity=(0, 0),
                        orientation=0.0, angular_velocity=0.0,
                        bounceable=True, restitution=0.1,
                    ),
                    team=0,
                ))

    def _init_ball(self):
        if self.ball_placement == "center":
            x = self.dimensions.stadium_length / 2
            y = self.dimensions.stadium_width / 2
            vx, vy = 0.0, 0.0
        else:
            x = random_float() * self.dimensions.stadium_length
            y = random_float() * self.dimensions.stadium_width
            vx = (2 * random_float() - 1) * self.physics.ball_max_speed * 0.3
            vy = (2 * random_float() - 1) * self.physics.ball_max_speed * 0.3

        self.ball = Ball(
            object=Object(
                radius=self.dimensions.ball_radius,
                position=(x, y), velocity=(vx, vy),
                orientation=0.0, angular_velocity=0.0,
                bounceable=True, restitution=1,
            )
        )

    def _init_game_state(self):
        self.game_state = GameState(
            game_time=0.0, score=[0, 0], possession=None,
            possession_time=[0.0, 0.0], last_kicker=None, last_kick_time=None,
        )
        self._done = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._init_players()
        self._init_ball()
        self._init_game_state()
        self._reward_components = [
            {"step": 0.0, "kick": 0.0, "proximity": 0.0,
             "ball_position": 0.0, "corner_avoidance": 0.0, "goal": 0.0}
            for _ in range(len(self.players))
        ]
        self.agents = list(self.possible_agents)

        return self._build_observations(), {agent: {} for agent in self.agents}

    def step(self, actions: Dict[str, np.ndarray]):
        if self._done:
            self.agents = []
            empty = {}
            return empty, empty, empty, empty, empty

        team0_actions = self._parse_team_actions(actions, team=0)

        if self._opponent_mode == "agents":
            team1_actions = self._parse_team_actions(actions, team=1)
        elif self._opponent_mode == "policy":
            team1_actions = self._get_opponent_actions()
        else:
            team1_actions = []

        per_player_rewards = self._physics_step(team0_actions + team1_actions)

        self.game_state.game_time += self.simulation.dt
        self._done = self.game_state.game_time >= self.simulation.max_game_time

        observations = self._build_observations()
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            team, idx = self._parse_agent_name(agent)
            player_idx = idx if team == 0 else self.team_size + idx
            rewards[agent] = float(per_player_rewards[player_idx])
            terminations[agent] = self._done
            truncations[agent] = False
            infos[agent] = {
                "score": list(self.game_state.score),
                "game_time": float(self.game_state.game_time),
                "possession": self.game_state.possession,
                "reward_components": dict(self._reward_components[player_idx]),
            }

        if self._done:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None

        if self._screen is None:
            pygame.init()
            pygame.display.init()
            self._scale = RENDER_SCALE
            self._width_pixels = int(
                (self.dimensions.stadium_length + 2 * self.dimensions.goal_depth + PADDING) * self._scale
            )
            self._height_pixels = int(
                (self.dimensions.stadium_width + 1.5 * PADDING) * self._scale
            )

            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((self._width_pixels, self._height_pixels))
                pygame.display.set_caption("2D Football Environment")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((self._width_pixels, self._height_pixels))

        self._draw_frame()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None

    # --- Observation building ---

    def _normalize_player_position(self, pos) -> Tuple[float, float]:
        return (2 * pos[0] / self.dimensions.stadium_length - 1,
                2 * pos[1] / self.dimensions.stadium_width - 1)

    def _normalize_ball_position(self, pos) -> Tuple[float, float]:
        gd = self.dimensions.goal_depth
        total_length = self.dimensions.stadium_length + 2 * gd
        return (2 * (pos[0] + gd) / total_length - 1,
                2 * pos[1] / self.dimensions.stadium_width - 1)

    def _normalize_velocity(self, vel, max_speed: float) -> Tuple[float, float]:
        return (vel[0] / max_speed, vel[1] / max_speed)

    def _player_obs(self, player: Player) -> np.ndarray:
        obj = player.object
        pos = self._normalize_player_position(obj.position)
        vel = self._normalize_velocity(obj.velocity, self.physics.player_max_speed)
        cos_o = np.cos(obj.orientation)
        sin_o = np.sin(obj.orientation)
        ang = obj.angular_velocity / self.physics.player_max_angular_speed
        return np.array([pos[0], pos[1], vel[0], vel[1], cos_o, sin_o, ang], dtype=np.float32)

    def _ball_obs(self) -> np.ndarray:
        obj = self.ball.object
        pos = self._normalize_ball_position(obj.position)
        vel = self._normalize_velocity(obj.velocity, self.physics.ball_max_speed)
        return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)

    def _mirror_player_obs(self, obs: np.ndarray) -> np.ndarray:
        pos = mirror_position((obs[0], obs[1]))
        vel = mirror_velocity((obs[2], obs[3]))
        cos_o = -obs[4]   # cos(θ+π) = -cos(θ)
        sin_o = -obs[5]   # sin(θ+π) = -sin(θ)
        ang = -obs[6]
        return np.array([pos[0], pos[1], vel[0], vel[1], cos_o, sin_o, ang], dtype=np.float32)

    def _mirror_ball_obs(self, obs: np.ndarray) -> np.ndarray:
        pos = mirror_position((obs[0], obs[1]))
        vel = mirror_velocity((obs[2], obs[3]))
        return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)

    def _can_kick(self, player: Player) -> bool:
        if distance(player.object, self.ball.object) > self.dimensions.player_radius + self.dimensions.ball_radius:
            return False
        if self.game_state.last_kicker is None or self.game_state.last_kicker != player:
            return True
        if self.game_state.last_kick_time is None:
            return True
        return (self.game_state.game_time - self.game_state.last_kick_time) > self.physics.kick_cooldown

    def _build_agent_obs(self, player_idx: int, team: int) -> np.ndarray:
        """Observation layout: [self(7), teammates(N*7), opponents(M*7), ball(4), ego_ball(3), kick(1), time(1)]"""
        team0_players = [p for p in self.players if p.team == 0]
        team1_players = [p for p in self.players if p.team == 1]
        ball_raw = self._ball_obs()

        if team == 0:
            my_players = team0_players
            opp_players = team1_players
            self_obs = self._player_obs(my_players[player_idx])
            teammate_obs = [self._player_obs(p) for j, p in enumerate(my_players) if j != player_idx]
            opponent_obs = [self._player_obs(p) for p in opp_players]
            ball = ball_raw
        else:
            my_players = team1_players
            opp_players = team0_players
            self_obs = self._mirror_player_obs(self._player_obs(my_players[player_idx]))
            teammate_obs = [self._mirror_player_obs(self._player_obs(p)) for j, p in enumerate(my_players) if j != player_idx]
            opponent_obs = [self._mirror_player_obs(self._player_obs(p)) for p in opp_players]
            ball = self._mirror_ball_obs(ball_raw)

        player_obj = my_players[player_idx].object
        ball_obj = self.ball.object
        dx = ball_obj.position[0] - player_obj.position[0]
        dy = ball_obj.position[1] - player_obj.position[1]
        cos_o = np.cos(player_obj.orientation)
        sin_o = np.sin(player_obj.orientation)
        diag = np.hypot(self.dimensions.stadium_length, self.dimensions.stadium_width)
        ego_ball = np.array([
            (dx * cos_o + dy * sin_o) / diag,
            (-dx * sin_o + dy * cos_o) / diag,
            np.hypot(dx, dy) / diag,
        ], dtype=np.float32)

        kick = np.array([1.0 if self._can_kick(my_players[player_idx]) else 0.0], dtype=np.float32)

        time_remaining = 1.0 - (self.game_state.game_time / self.simulation.max_game_time)

        parts = [self_obs] + teammate_obs + opponent_obs + [ball, ego_ball, kick, np.array([time_remaining], dtype=np.float32)]
        return np.clip(np.concatenate(parts), -1.0, 1.0)

    def _build_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for agent in self.agents:
            team, idx = self._parse_agent_name(agent)
            obs[agent] = self._build_agent_obs(idx, team)
        return obs

    def _build_opponent_observations(self) -> Dict[str, np.ndarray]:
        return {
            vname: self._build_agent_obs(i, team=1)
            for i, vname in enumerate(self._opponent_virtual_agents)
        }

    # --- Action handling ---

    def _denormalize_action(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            float(action[0]) * self.physics.player_max_acceleration,
            float(action[1]) * self.physics.player_max_angular_acceleration,
            float(action[2]) * self.physics.player_max_kick_impulse,
            float(action[3]) * np.pi,
        )

    def _parse_team_actions(self, actions: Dict[str, np.ndarray], team: int) -> list:
        prefix = f"team{team}_player"
        return [self._denormalize_action(actions[f"{prefix}{i}"]) for i in range(self.team_size)]

    def _get_opponent_actions(self) -> list:
        opp_obs = self._build_opponent_observations()
        opp_actions_raw = self._opponent_policy(opp_obs)

        result = []
        for i in range(self.team_size):
            vname = self._opponent_virtual_agents[i]
            mirrored = mirror_action(np.asarray(opp_actions_raw[vname], dtype=np.float32))
            result.append(self._denormalize_action(mirrored))
        return result

    # --- Physics step ---

    def _physics_step(self, all_actions: list) -> List[float]:
        goal_min_y = (self.dimensions.stadium_width - self.dimensions.goal_width) / 2
        goal_max_y = goal_min_y + self.dimensions.goal_width

        per_player_rewards = [STEP_REWARD] * len(self.players)
        for rc in self._reward_components:
            rc["step"] += STEP_REWARD

        prev_ball_x = self.ball.object.position[0]
        prev_distances = [distance(player.object, self.ball.object) for player in self.players]

        L, W = self.dimensions.stadium_length, self.dimensions.stadium_width
        prev_corner_dists = []
        for player in self.players:
            px, py = player.object.position
            prev_corner_dists.append(min(
                np.hypot(px, py), np.hypot(px, W - py),
                np.hypot(L - px, py), np.hypot(L - px, W - py),
            ))

        for i, player in enumerate(self.players):
            acceleration, angular_acceleration, _, _ = all_actions[i]
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
                min_length=0, max_length=self.dimensions.stadium_length,
                min_width=0, max_width=self.dimensions.stadium_width,
                goal_min_y=goal_min_y, goal_max_y=goal_max_y,
                goal_depth=self.dimensions.goal_depth,
                allow_goal_entry=False,
            )

        new_objects = resolve_collisions([player.object for player in self.players])
        for i, player in enumerate(self.players):
            player.object = new_objects[i]

        ball_acceleration = np.array([0.0, 0.0])
        possession = None
        sum_radii = self.dimensions.player_radius + self.dimensions.ball_radius

        for i, player in enumerate(self.players):
            _, _, kicking_force, kicking_angle = all_actions[i]
            dist_to_ball = distance(player.object, self.ball.object)

            if dist_to_ball <= sum_radii and (
                self.game_state.last_kicker is None
                or self.game_state.last_kicker != player
                or (self.game_state.game_time - self.game_state.last_kick_time) > self.physics.kick_cooldown
            ):
                kick_direction = player.object.orientation + kicking_angle
                kick_x = np.cos(kick_direction) * kicking_force
                kick_y = np.sin(kick_direction) * kicking_force

                ball_acceleration += np.array([kick_x, kick_y])
                per_player_rewards[i] += KICK_REWARD
                self._reward_components[i]["kick"] += KICK_REWARD
                possession = player.team
                self.game_state.last_kicker = player
                self.game_state.last_kick_time = self.game_state.game_time

        self.ball.object.act(
            acceleration=ball_acceleration, angular_acceleration=0,
            dt=self.simulation.dt,
            max_speed=self.physics.ball_max_speed, max_angular_speed=None,
            friction_factor=self.physics.ball_friction_factor, angular_friction_factor=1,
            min_length=0, max_length=self.dimensions.stadium_length,
            min_width=0, max_width=self.dimensions.stadium_width,
            goal_min_y=goal_min_y, goal_max_y=goal_max_y,
            goal_depth=self.dimensions.goal_depth,
            instant=True,
            allow_goal_entry=True,
        )

        self.game_state.possession = possession
        if possession is not None:
            self.game_state.possession_time[possession] += self.simulation.dt

        ball_x, ball_y = self.ball.object.position
        ball_x_delta = (ball_x - prev_ball_x) / self.dimensions.stadium_length

        half_diag = np.hypot(L, W) / 2

        for i, player in enumerate(self.players):
            curr_dist = distance(player.object, self.ball.object)
            proximity_delta = (prev_distances[i] - curr_dist) / self.dimensions.stadium_length
            proximity_reward = proximity_delta * PROXIMITY_REWARD_SCALE
            per_player_rewards[i] += proximity_reward

            if player.team == 0:
                bp_reward = ball_x_delta * BALL_POSITION_REWARD_SCALE
            else:
                bp_reward = -ball_x_delta * BALL_POSITION_REWARD_SCALE
            per_player_rewards[i] += bp_reward

            px, py = player.object.position
            curr_corner_dist = min(
                np.hypot(px, py), np.hypot(px, W - py),
                np.hypot(L - px, py), np.hypot(L - px, W - py),
            )
            corner_delta = (curr_corner_dist - prev_corner_dists[i]) / half_diag
            corner_reward = corner_delta * CORNER_AVOIDANCE_REWARD_SCALE
            per_player_rewards[i] += corner_reward

            self._reward_components[i]["proximity"] += proximity_reward
            self._reward_components[i]["ball_position"] += bp_reward
            self._reward_components[i]["corner_avoidance"] += corner_reward

        is_in_goal_y_range = goal_min_y <= ball_y <= goal_max_y

        if ball_x < 0 and is_in_goal_y_range:
            self._handle_goal(scoring_team=1, per_player_rewards=per_player_rewards)
        elif ball_x > self.dimensions.stadium_length and is_in_goal_y_range:
            self._handle_goal(scoring_team=0, per_player_rewards=per_player_rewards)

        return per_player_rewards

    def _handle_goal(self, scoring_team: int, per_player_rewards: List[float]):
        self.game_state.score[scoring_team] += 1
        self._init_players()
        self._init_ball()
        self.game_state.possession = None
        self.game_state.last_kicker = None
        self.game_state.last_kick_time = None

        for i, player_team in enumerate(
            [0] * self.team_size + ([1] * self.team_size if self._has_opponent else [])
        ):
            goal_reward = 1.0 if player_team == scoring_team else -1.0
            per_player_rewards[i] += goal_reward
            self._reward_components[i]["goal"] += goal_reward

    # --- Rendering ---

    def _draw_frame(self):
        scale = self._scale
        self._screen.fill(GREEN)

        def draw_rectangle(x, y, length, width, color=WHITE):
            pygame.draw.rect(self._screen, color, pygame.Rect(x, y, length, width), 2)

        def draw_circle(x, y, radius, color=WHITE):
            pygame.draw.circle(self._screen, color, (int(x), int(y)), int(radius))

        pygame.draw.rect(self._screen, BLACK, (0, 0, self._width_pixels, 0.375 * PADDING * scale))

        font = pygame.font.SysFont("Consolas", int(0.25 * PADDING * scale))
        score_text = (
            f"A {self.game_state.score[0]} - {self.game_state.score[1]} B  |  "
            f"Time: {self.game_state.game_time:.1f} / {self.simulation.max_game_time}"
        )
        text_surface = font.render(score_text, True, WHITE)
        self._screen.blit(text_surface, (
            (self._width_pixels - text_surface.get_width()) // 2,
            0.375 / 2 * PADDING * scale - text_surface.get_height() // 2,
        ))

        field_length_px = self.dimensions.stadium_length * scale
        field_width_px = self.dimensions.stadium_width * scale
        offset_x = (self._width_pixels - field_length_px) // 2
        offset_y = (self._height_pixels - field_width_px) // 2

        draw_rectangle(offset_x, offset_y, field_length_px, field_width_px)

        center_x = offset_x + field_length_px // 2
        center_y = offset_y + field_width_px // 2
        pygame.draw.line(self._screen, WHITE, (center_x, offset_y), (center_x, offset_y + field_width_px), 2)
        pygame.draw.circle(self._screen, WHITE, (center_x, center_y), int(self.dimensions.center_circle_radius * scale), 2)

        box_length_px = self.dimensions.penalty_area_length * scale
        box_width_px = self.dimensions.penalty_area_width * scale
        goal_area_length_px = self.dimensions.goal_area_length * scale
        goal_area_width_px = self.dimensions.goal_area_width * scale

        penalty_top_y = offset_y + (field_width_px - box_width_px) / 2
        goal_top_y = offset_y + (field_width_px - goal_area_width_px) / 2

        draw_rectangle(offset_x, penalty_top_y, box_length_px, box_width_px)
        draw_rectangle(offset_x + field_length_px - box_length_px, penalty_top_y, box_length_px, box_width_px)
        draw_rectangle(offset_x, goal_top_y, goal_area_length_px, goal_area_width_px)
        draw_rectangle(offset_x + field_length_px - goal_area_length_px, goal_top_y, goal_area_length_px, goal_area_width_px)

        penalty_spot_radius_px = self.dimensions.penalty_spot_radius * scale
        left_penalty_x = offset_x + self.dimensions.penalty_spot_distance * scale
        right_penalty_x = offset_x + field_length_px - self.dimensions.penalty_spot_distance * scale
        draw_circle(left_penalty_x, center_y, penalty_spot_radius_px)
        draw_circle(right_penalty_x, center_y, penalty_spot_radius_px)

        goal_depth_x = offset_x - self.dimensions.goal_depth * scale
        goal_depth_px = self.dimensions.goal_depth * scale
        goal_post_width_px = self.dimensions.goal_width * scale
        goal_post_top_y = offset_y + (field_width_px - goal_post_width_px) / 2
        draw_rectangle(goal_depth_x + 1, goal_post_top_y, goal_depth_px, goal_post_width_px, color=RED)
        draw_rectangle(offset_x + field_length_px - 1, goal_post_top_y, goal_depth_px, goal_post_width_px, color=BLUE)

        for player in self.players:
            px = offset_x + player.object.position[0] * scale
            py = offset_y + player.object.position[1] * scale
            color = RED if player.team == 0 else BLUE
            draw_circle(px, py, self.dimensions.player_radius * scale, color)

            dot_offset = self.dimensions.player_radius * scale * 0.5
            angle = player.object.orientation
            draw_circle(px + np.cos(angle) * dot_offset, py + np.sin(angle) * dot_offset,
                        self.dimensions.player_radius * scale * 0.3, WHITE)

        ball_x = offset_x + self.ball.object.position[0] * scale
        ball_y = offset_y + self.ball.object.position[1] * scale
        draw_circle(ball_x, ball_y, self.dimensions.ball_radius * scale, YELLOW)

        if self.render_mode == "human":
            pygame.display.flip()

    # --- Utilities ---

    @staticmethod
    def _parse_agent_name(agent: str) -> Tuple[int, int]:
        parts = agent.split("_")
        return int(parts[0].replace("team", "")), int(parts[1].replace("player", ""))
