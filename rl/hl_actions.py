from .reward import rewards

# Perform a high-level action
def perform_action(self, action, **kwargs):
    reward = 0
    if action == "Pass":
        self.pass_ball()
        reward = rewards["Pass"]
    elif action == "Shoot":
        self.shoot()
        reward = rewards["Shoot"]
    elif action == "Follow":
        self.follow_ball()
        reward = rewards["Follow"]
    elif action == "Tackle":
        self.tackle()
        reward = rewards["Tackle"]
    elif action == "Accelerate":
        self.accelerate()
        reward = rewards["Accelerate"]
    elif action == "ChangeOrientation":
        delta_angle = kwargs.get("delta_angle", 0)  # Default to 0 if not provided
        self.change_orientation(delta_angle)
        reward = rewards["ChangeOrientation"]
    else:
        print(f"Unknown action: {action}")
    
    # Return the reward for the action
    return reward

def pass_ball(self):
    #Pass the ball to a teammate
    if self.has_ball:
        print(f"Agent {self.agent_id} is passing the ball.")
        self.has_ball = False
    else:
        print(f"Agent {self.agent_id} cannot pass: they do not have the ball.")

def shoot(self):
    #Shoot the ball towards the goal
    if self.has_ball:
        print(f"Agent {self.agent_id} is shooting.")
        self.has_ball = False
    else:
        print(f"Agent {self.agent_id} cannot shoot: they do not have the ball.")

def follow_ball(self):
    #Follow the ball or opponent with the ball
    print(f"Agent {self.agent_id} is following the ball.")

def tackle(self):
    #Attempt to tackle the opponent with the ball
    print(f"Agent {self.agent_id} is attempting a tackle.")

def accelerate(self):
    #Accelerate in the current orientation
    print(f"Agent {self.agent_id} is accelerating.")

def change_orientation(self, delta_angle):
    #Change orientation using joystick-like control.
    self.orientation = (self.orientation + delta_angle) % 360
    print(f"Agent {self.agent_id} is changing orientation to {self.orientation} degrees.")
