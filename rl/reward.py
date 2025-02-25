#Rewards for Actions and Events
rewards = {
    "Pass": 0.5,  #Reward for a successful pass
    "Shoot": 0.8,  #Reward for attempting a shot
    "Follow": 0.2,  #Reward for following the ball
    "Tackle": 1.0,  #Reward for a successful tackle
    "Accelerate": 0.1,  #Reward for accelerating
    "ChangeOrientation": 0.0,  #No direct reward for changing orientation
    "GoalScored": 10.0,  #Reward for scoring a goal
    "BallIntercepted": 1.0,  #Reward for intercepting the ball
    "ConcedeGoal": -5.0,  #Penalty for conceding a goal
    "LostBall": -0.5,  #Penalty for losing the ball
}
