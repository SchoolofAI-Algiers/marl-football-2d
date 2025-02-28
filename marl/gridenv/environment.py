import pygame
import random
from config import *


class FootballEnv:
    def __init__(self):

        self.grid_size = (4, 7)
        self.start_positions = [(1, 4), (2, 2)] 
        self.goals = {"A": [(1, 0), (2, 0)], "B": [(1, 6), (2, 6)]}  
        self.non_accessible_cells = [(0, 0), (0, 6), (3, 0), (3, 6)]

        self.done = False

        self.reset()

    def _get_state(self):
        return (self.positions["A"], self.positions["B"], self.ball_owner)

    def reset(self):
        """
        Resets the environment:
        - Resets the initial positions of players
        - Resets ball ownership to a random player (aka team)
        """
        self.positions = {"A": self.start_positions[0], "B": self.start_positions[1]}
        self.ball_owner = random.choice(["A", "B"])
        return self._get_state()

    def step(self, actions):
        """
        One simulation step:
        - Applies actions to players 
        - Updates positions and ball dynamics
        - Checks for goals and updates score/rewards
        - Returns (state, rewards [, done: TO INCLUDE, info: TO INCLUDE])
        """

        moves = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1), "STAY": (0, 0)}
        new_positions = {}

        for agent in actions:
            new_pos = (self.positions[agent][0] + moves[actions[agent]][0],
                    self.positions[agent][1] + moves[actions[agent]][1])
            
            # Check if the new position is within bounds and not in a non-accessible cell
            if (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1] and
                new_pos not in self.non_accessible_cells):
                new_positions[agent] = new_pos
            else:
                new_positions[agent] = self.positions[agent]  # Stay in place if move is invalid

        # Check for collision (one agent moving into the other's position)
        if new_positions["A"] == self.positions["B"] and new_positions["B"] == self.positions["B"]:
            # A tries to move into B's position, possession goes to B
            self.ball_owner = "B"
            new_positions["A"] = self.positions["A"]  # A's move is canceled

        elif new_positions["B"] == self.positions["A"] and new_positions["A"] == self.positions["A"]:
            # B tries to move into A's position, possession goes to A
            self.ball_owner = "A"
            new_positions["B"] = self.positions["B"]  # B's move is canceled

        # Update positions
        self.positions = new_positions

        # Check if an agent has scored
        for agent in self.positions:
            if self.positions[agent] in self.goals[agent] and self.ball_owner == agent:
                reward = {"A": 1 if agent == "A" else -1, "B": 1 if agent == "B" else -1}
                self.reset()
                return self._get_state(), reward

        return self._get_state(), {"A": 0, "B": 0}

    

    def render(self):
        """
        Renders a grid football stadium with:
        - Green background, white boundary lines, blue nets (Gray for non accessible cells)
        - Labeled colored circles representing players 
        - A ball 
        """
        if not hasattr(self, 'screen'):
            pygame.init()
            self.soccer_ball_img = pygame.image.load("./assets/soccer_ball.png") 
            self.soccer_ball_img = pygame.transform.scale(self.soccer_ball_img, (20, 20)) 
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Football Environment")


        self.screen.fill(GREEN) 
        
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
                if (row, col) in self.non_accessible_cells:
                    pygame.draw.rect(self.screen, GRAY, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        for row in range(1, self.grid_size[0]):
            pygame.draw.line(self.screen, WHITE, (0, row * CELL_SIZE), (WIDTH, row * CELL_SIZE), 2)
        for col in range(1, self.grid_size[1]):
            pygame.draw.line(self.screen, WHITE, (col * CELL_SIZE, 0), (col * CELL_SIZE, HEIGHT), 2)

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