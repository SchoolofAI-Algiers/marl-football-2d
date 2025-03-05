import pygame
import random

from config import (
    ASSETS_PATH, 
    GRID_SIZES, CELL_SIZE, DEFAULT_SIZE,
    GREEN, BLUE, WHITE, GRAY, A_COLOR, B_COLOR
)

class FootballEnv:
    
    def __init__(self, size: str = DEFAULT_SIZE):
        self.grid_size = GRID_SIZES[size]
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]
        self._make_goals()
        self._init_positions()
        self.ball_owner = random.choice(["A", "B"])
        self.action_space = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "STAY"]
        
        self.score = {"A": 0, "B": 0}

        self.done = False
        self.timestep = 0

        self.reset()
            
    def _make_goals(self):
        """
        Initializes the goals, non-accessible cells
        """
        if self.height % 2 == 0:
            goal_x = [i for i in range(self.height//2 - 1, self.height//2 + 1)]
            self.goals = {"A": [(i, 0) for i in goal_x],
                          "B": [(i, self.width - 1) for i in goal_x]}
        else:
            goal_x = [i for i in range(self.height//2-1, self.height//2 + 2)]
            self.goals = {"A": [(i, 0) for i in goal_x],
                          "B": [(i, self.width - 1) for i in goal_x]}
            
        self.non_accessible_cells = []
        self.non_accessible_cells += [(i, 0) for i in range(self.height)]
        self.non_accessible_cells += [(i, self.width - 1) for i in range(self.height)]
        for goal in self.goals["A"] + self.goals["B"]:
            self.non_accessible_cells.remove(goal)
      
    def _init_positions(self):
        """
        Initializes the players' positions
        """
        
        self.start_positions = [
            (random.randint(0, self.height - 1), random.randint((self.width - 1)//2 + 1, self.width - 2)),
            (random.randint(0, self.height - 1), random.randint(1, (self.width - 1)//2 - 1))
        ]
        
        self.positions = {"A": self.start_positions[0], "B": self.start_positions[1]}
          
    def _get_state(self):
        return (self.positions["A"], self.positions["B"], self.ball_owner, self.score, self.timestep)

    def reset(self, on_goal=False, conceeding_team=None):
        """
        Resets the environment:
        - Resets the initial positions of players
        - Resets ball ownership to a random player (aka team)
        """
        self._make_goals()
        self._init_positions()
        if not on_goal:
            self.ball_owner = random.choice(["A", "B"])
            self.score = {"A": 0, "B": 0}
            self.timestep = 0
        else:
            self.ball_owner = conceeding_team
        return self._get_state()

    def step(self, actions):
        """
        One simulation step:
        - Applies actions to players 
        - Updates positions and ball dynamics
        - Checks for goals and updates score/rewards
        - Returns (state, rewards [, done: TO INCLUDE, info: TO INCLUDE])
        """
        self.timestep += 1

        moves = {
            "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
            "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1),
            "STAY": (0, 0)
        }

        new_positions = {}

        for agent in actions:
            new_pos = (self.positions[agent][0] + moves[actions[agent]][0],
                    self.positions[agent][1] + moves[actions[agent]][1])

            # Check if the new position is valid
            if (0 <= new_pos[0] < self.grid_size[0] and
                0 <= new_pos[1] < self.grid_size[1] and
                new_pos not in self.non_accessible_cells):
                new_positions[agent] = new_pos
            else:
                new_positions[agent] = self.positions[agent]  # Stay in place if move is invalid

        # Collision handling: If a player moves into the other’s square, possession changes
        if new_positions["A"] == new_positions["B"]:
            current_ball_owner = self.ball_owner
            if current_ball_owner == "A":
                self.ball_owner = "B"
                # new_positions["A"] = self.positions["A"]  # A's move is canceled
            else:
                self.ball_owner = "A"
                # new_positions["B"] = self.positions["B"]

        # Update positions
        self.positions = new_positions

        # Check if an agent has scored
        for agent in self.positions:
            other_agent = "B" if agent == "A" else "A"
            if self.positions[agent] in self.goals[agent] and self.ball_owner == agent:
                reward = {"A": 1 if agent == "A" else -1, "B": 1 if agent == "B" else -1}
                self.score[agent] += 1
                return self._get_state(), reward, self.done, {'goal': True, 'conceeding_team': other_agent}
            if self.positions[agent] in self.goals[other_agent] and self.ball_owner == agent:
                reward = {"A": 1 if agent == "B" else -1, "B": 1 if agent == "A" else -1}
                self.score[other_agent] += 1
                return self._get_state(), reward, self.done, {'goal': True, 'conceeding_team': agent}

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
        font = pygame.font.SysFont("Consolas", 30)
        score_text = f"A {self.score['A']} - {self.score['B']} B  |  Timestep: {self.timestep}"
        text_surface = font.render(score_text, True, WHITE)

        # Center and display text
        self.screen.blit(text_surface, ((self.width * CELL_SIZE - text_surface.get_width()) // 2, self.height * CELL_SIZE + CELL_SIZE // 2 - text_surface.get_height() // 2))
        
        # Draw goals 
        for goal in self.goals["A"]:
            pygame.draw.rect(self.screen, BLUE, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)  
            # Square stripes for the goal net
            for i in range(0, CELL_SIZE, 20):  
                pygame.draw.line(self.screen, WHITE, (goal[1] * CELL_SIZE + i, goal[0] * CELL_SIZE), 
                                 (goal[1] * CELL_SIZE + i, goal[0] * CELL_SIZE + CELL_SIZE), 2)
                pygame.draw.line(self.screen, WHITE, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE + i), 
                                 (goal[1] * CELL_SIZE + CELL_SIZE, goal[0] * CELL_SIZE + i), 2)

        for goal in self.goals["B"]:
            pygame.draw.rect(self.screen, BLUE, (goal[1] * CELL_SIZE, goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)  
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

        # Draw players
        a_x, a_y = self.positions["A"]
        b_x, b_y = self.positions["B"]

        # Player A 
        pygame.draw.circle(self.screen, A_COLOR, (a_y * CELL_SIZE + CELL_SIZE // 2, a_x * CELL_SIZE + CELL_SIZE // 2), 15)
        if self.ball_owner == "A":
            self.screen.blit(self.soccer_ball_img, (a_y * CELL_SIZE + 25, a_x * CELL_SIZE + 65))  # Add Ball 

        # Player B 
        pygame.draw.circle(self.screen, B_COLOR, (b_y * CELL_SIZE + CELL_SIZE // 2, b_x * CELL_SIZE + CELL_SIZE // 2), 15)
        if self.ball_owner == "B":
            self.screen.blit(self.soccer_ball_img, (b_y * CELL_SIZE + 25, b_x * CELL_SIZE + 65))  # Add Ball 

        # Draw player names inside the circle
        font = pygame.font.SysFont("Arial", 20)
        text_A = font.render("A", True, WHITE)
        text_B = font.render("B", True, WHITE)
        self.screen.blit(text_A, (a_y * CELL_SIZE + CELL_SIZE // 2 - text_A.get_width() // 2, a_x * CELL_SIZE + CELL_SIZE // 2 - text_A.get_height() // 2))
        self.screen.blit(text_B, (b_y * CELL_SIZE + CELL_SIZE // 2 - text_B.get_width() // 2, b_x * CELL_SIZE + CELL_SIZE // 2 - text_B.get_height() // 2))

        # Refresh the screen
        pygame.display.flip()
