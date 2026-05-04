import pygame
import random

from config import (
    ASSETS_PATH, 
    GRID_SIZES, CELL_SIZE, DEFAULT_SIZE,
    GREEN, WHITE, GRAY, A_COLOR, B_COLOR,
    MAX_TIMESTEP
)

class FootballEnv:
    
    def __init__(self, team_size: int = 2, size: str = DEFAULT_SIZE):
        self.grid_size = GRID_SIZES[size]
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]
        self.team_size = team_size
        self._make_goals()
        self._init_positions()
        
        self.ball_owner = random.choice(["A", "B"]), random.randint(0, self.team_size - 1)
        self.ball_position = self.positions[self.ball_owner[0]][self.ball_owner[1]]
        self.ball_direction = None  # Ball is stationary at the start
        self.action_space = [
            "N", "S", "E", "W", "NE", "NW", "SE", "SW", "STAY",
            "KICK_N", "KICK_S", "KICK_E", "KICK_W", "KICK_NE", "KICK_NW", "KICK_SE", "KICK_SW"
        ]
        
        self.score = {"A": 0, "B": 0}
        self.done = False
        self.timestep = 0
        self.max_timestep = MAX_TIMESTEP

        self.reset()
    
    def _make_goals(self):
        """Initializes goal areas and non-accessible cells."""
        if self.height % 2 == 0:
            goal_x = [i for i in range(self.height//2 - 1, self.height//2 + 1)]
        else:
            goal_x = [i for i in range(self.height//2-1, self.height//2 + 2)]
        
        self.goals = {
            "A": [(i, self.width - 1) for i in goal_x],
            "B": [(i, 0) for i in goal_x]
        }

        self.non_accessible_cells = [(i, 0) for i in range(self.height)]
        self.non_accessible_cells += [(i, self.width - 1) for i in range(self.height)]
        for goal in self.goals["A"] + self.goals["B"]:
            self.non_accessible_cells.remove(goal)

    def _init_positions(self):
        """Initializes player positions for both teams."""
        self.positions = {"A": [], "B": []}
        
        for _ in range(self.team_size):
            self.positions["A"].append((random.randint(0, self.height - 1), random.randint((self.width - 1)//2 + 1, self.width - 2)))
            self.positions["B"].append((random.randint(0, self.height - 1), random.randint(1, (self.width - 1)//2 - 1)))

    def _get_state(self):
        return (self.positions, self.ball_owner, self.score, self.timestep)

    def reset(self, on_goal=False, conceding_team=None):
        """Resets environment to initial state."""
        self._make_goals()
        self._init_positions()
        if not on_goal:
            self.ball_owner = (random.choice(["A", "B"]), random.randint(0, self.team_size - 1))
            self.score = {"A": 0, "B": 0}
            self.timestep = 0
        else:
            self.ball_owner = (conceding_team, random.randint(0, self.team_size - 1))

        self.ball_position = self.positions[self.ball_owner[0]][self.ball_owner[1]]
        self.ball_direction = None
        
        return self._get_state()

    def step(self, actions):
        """Processes actions for all players."""
        self.timestep += 1
        if self.timestep >= self.max_timestep:
            self.done = True

        moves = {
            "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
            "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1),
            "STAY": (0, 0)
        }
        
        kicks = {
            "KICK_N": (-1, 0), "KICK_S": (1, 0), "KICK_E": (0, 1), "KICK_W": (0, -1),
            "KICK_NE": (-1, 1), "KICK_NW": (-1, -1), "KICK_SE": (1, 1), "KICK_SW": (1, -1)
        }

        new_positions = {"A": [], "B": []}

        # Move players
        for team in ["A", "B"]:
            for i in range(self.team_size):
                action = actions[team][i]
                current_pos = self.positions[team][i]

                if self.ball_owner == (team, i):
                    if action in kicks:
                        self.ball_direction = kicks[action]
                        self.ball_owner = None
                        new_positions[team].append(current_pos)  
                    else:
                        new_pos = (current_pos[0] + moves[action][0], current_pos[1] + moves[action][1])

                        if (0 <= new_pos[0] < self.grid_size[0] and
                            0 <= new_pos[1] < self.grid_size[1] and
                            new_pos not in self.non_accessible_cells):
                            new_positions[team].append(new_pos)
                            self.ball_position = new_pos
                        else:
                            new_positions[team].append(current_pos)  
                            self.ball_position = current_pos  

                elif action in kicks:
                    new_positions[team].append(current_pos)

                else:
                    new_pos = (current_pos[0] + moves[action][0], current_pos[1] + moves[action][1])

                    if (0 <= new_pos[0] < self.grid_size[0] and
                        0 <= new_pos[1] < self.grid_size[1] and
                        new_pos not in self.non_accessible_cells):
                        new_positions[team].append(new_pos)
                    else:
                        new_positions[team].append(current_pos) 

        # Move the ball if it is free
        if self.ball_direction:
            new_ball_pos = (self.ball_position[0] + self.ball_direction[0],
                            self.ball_position[1] + self.ball_direction[1])

            # Check if the new ball position is valid
            if (0 <= new_ball_pos[0] < self.grid_size[0] and
                0 <= new_ball_pos[1] < self.grid_size[1] and
                new_ball_pos not in self.non_accessible_cells):
                self.ball_position = new_ball_pos
            else:
                self.ball_direction = None  # Stop ball if it hits a wall

        # Collision handling (Ball Stealing)
        if self.ball_owner:
            ball_owner_team, ball_owner_idx = self.ball_owner
            ball_owner_pos = new_positions[ball_owner_team][ball_owner_idx]
            other_team = "B" if ball_owner_team == "A" else "A"
            
            # Find all opponents trying to steal
            stealers = [i for i, pos in enumerate(new_positions[other_team]) if pos == ball_owner_pos]

            if stealers:
                if random.choice([True, False]):  # 50% chance of stealing
                    self.ball_owner = (other_team, random.choice(stealers))
                    self.ball_direction = None  # Stop autonomous movement

        # If the ball is free, allow claiming
        elif self.ball_owner is None:
            claimers = []  # List of players stepping on the ball

            for team in ["A", "B"]:
                for i, pos in enumerate(new_positions[team]):
                    if pos == self.ball_position:
                        claimers.append((team, i))

            if claimers:
                self.ball_owner = random.choice(claimers)  # Randomly choose one player to claim the ball
                self.ball_direction = None  # Stop autonomous movement

        # Update positions
        self.positions = new_positions

        # Check if the ball enters a goal
        if self.ball_position in self.goals["A"]:
            reward = {"A": -1, "B": 1}
            self.score["B"] += 1
            return self._get_state(), reward, self.done, {'goal': True, 'conceding_team': "A"}
        elif self.ball_position in self.goals["B"]:
            reward = {"A": 1, "B": -1}
            self.score["A"] += 1
            return self._get_state(), reward, self.done, {'goal': True, 'conceding_team': "B"}

        return self._get_state(), {"A": 0, "B": 0}, self.done, {'goal': False}

    def render(self):
        """
        Renders a grid football stadium with:
        - Green background, white boundary lines, blue nets (Gray for non accessible cells)
        - Labeled colored circles representing players 
        - A ball 
        """
        if not hasattr(self, 'screen'):
            pygame.init()
            self.soccer_ball_img = pygame.image.load(f"{ASSETS_PATH}/soccer_ball.png") 
            self.soccer_ball_img = pygame.transform.scale(self.soccer_ball_img, (20, 20)) 
            self.screen = pygame.display.set_mode((self.width * CELL_SIZE, self.height * CELL_SIZE + CELL_SIZE))
            pygame.display.set_caption("Football Environment")

        self.screen.fill(GREEN)
        
        # Draw scoreboard background
        pygame.draw.rect(self.screen, (0, 0, 0), (0, self.height * CELL_SIZE, self.width * CELL_SIZE, CELL_SIZE))

        # Render scoreboard text
        font = pygame.font.SysFont("Consolas", CELL_SIZE // 4)
        score_text = f"A {self.score['A']} - {self.score['B']} B  |  Timestep: {self.timestep} / {self.max_timestep}"
        text_surface = font.render(score_text, True, WHITE)

        # Center and display text
        self.screen.blit(text_surface, ((self.width * CELL_SIZE - text_surface.get_width()) // 2, self.height * CELL_SIZE + CELL_SIZE // 2 - text_surface.get_height() // 2))
        
        # Draw goals 
        for goal in self.goals["A"]:
            pygame.draw.rect(self.screen, GREEN, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)  
            # Square stripes for the goal net
            for i in range(0, CELL_SIZE, 20):  
                pygame.draw.line(self.screen, WHITE, (goal[1] * CELL_SIZE + i, goal[0] * CELL_SIZE), 
                                 (goal[1] * CELL_SIZE + i, goal[0] * CELL_SIZE + CELL_SIZE), 2)
                pygame.draw.line(self.screen, WHITE, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE + i), 
                                 (goal[1] * CELL_SIZE + CELL_SIZE, goal[0] * CELL_SIZE + i), 2)

        for goal in self.goals["B"]:
            pygame.draw.rect(self.screen, GREEN, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)  
            # Square stripes for the goal net
            for i in range(0, CELL_SIZE, 20):  
                pygame.draw.line(self.screen, WHITE, (goal[1] * CELL_SIZE + i, goal[0] * CELL_SIZE), 
                                 (goal[1] * CELL_SIZE + i, goal[0] * CELL_SIZE + CELL_SIZE), 2)
                pygame.draw.line(self.screen, WHITE, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE + i), 
                                 (goal[1] * CELL_SIZE + CELL_SIZE, goal[0] * CELL_SIZE + i), 2)

        # Draw grid lines and non-accessible cells
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                if (row, col) in self.non_accessible_cells and (row, col) not in self.goals["A"] and (row, col) not in self.goals["B"]:
                    pygame.draw.rect(self.screen, GRAY, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        for row in range(1, self.grid_size[0]):
            pygame.draw.line(self.screen, WHITE, (0, row * CELL_SIZE), (self.width * CELL_SIZE, row * CELL_SIZE), 2)
        for col in range(1, self.grid_size[1]):
            pygame.draw.line(self.screen, WHITE, (col * CELL_SIZE, 0), (col * CELL_SIZE, self.height * CELL_SIZE), 2)
        
        font = pygame.font.SysFont("Arial", 15)
        offset = 20
        offsets = [(0, 0), (0, offset), (0, -offset), (offset, 0), (-offset, 0), (offset, offset), (-offset, -offset), (offset, -offset), (-offset, offset)]
        
        # Draw free ball with a direction line
        if self.ball_owner is None:
            x, y = self.ball_position
            if self.ball_direction:
                line_length = 15
                pygame.draw.line(self.screen, WHITE, (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), (y * CELL_SIZE + CELL_SIZE // 2 + self.ball_direction[1] * line_length, x * CELL_SIZE + CELL_SIZE // 2 + self.ball_direction[0] * line_length), 3)
            self.screen.blit(self.soccer_ball_img, (y * CELL_SIZE + CELL_SIZE // 2 - 10, x * CELL_SIZE + CELL_SIZE // 2 - 10))
        
        # Draw players
        for i in range(self.team_size):
            x, y = self.positions["A"][i]
            px, py = random.choice(offsets)
            color = A_COLOR
            pygame.draw.circle(self.screen, color, (y * CELL_SIZE + CELL_SIZE // 2 + px, x * CELL_SIZE + CELL_SIZE // 2 + py), 15)
            if self.ball_owner == ("A", i):
                self.screen.blit(self.soccer_ball_img, (y * CELL_SIZE + CELL_SIZE // 4 - 5 + px, x * CELL_SIZE + CELL_SIZE * 3 // 8 + py))
                
            text = font.render(f"A{i}", True, WHITE)
            self.screen.blit(text, (y * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2 + px, x * CELL_SIZE + CELL_SIZE // 2 - text.get_height() // 2 + py))
        
        for i in range(self.team_size):
            x, y = self.positions["B"][i]
            px, py = random.choice(offsets)
            color = B_COLOR
            pygame.draw.circle(self.screen, color, (y * CELL_SIZE + CELL_SIZE // 2 + px, x * CELL_SIZE + CELL_SIZE // 2 + py), 15)
            if self.ball_owner == ("B", i):
                self.screen.blit(self.soccer_ball_img, (y * CELL_SIZE + CELL_SIZE // 2 + 5 + px, x * CELL_SIZE + CELL_SIZE * 3 // 8 + py))
                
            text = font.render(f"B{i}", True, WHITE)
            self.screen.blit(text, (y * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2 + px, x * CELL_SIZE + CELL_SIZE // 2 - text.get_height() // 2 + py))

        # Refresh the screen
        pygame.display.flip()
