import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pygame
import math
import sys
import random

numOfRows = 6
numOfColumns = 7
RED = (255 , 0 , 0)
GREEN = (0, 200, 0)
BLUE = (0 , 0 , 200)
BLACK = (0 , 0 , 0)
YELLOW = (255 , 255 , 0)
PURPLE = (255 , 0 , 255)
GREY = 	(192,192,192)

# PLAYER = 0
# AGENT = 1
PLAYER_PIECE = 1
AGENT_PIECE = 2
EMPTY = 0
Window_LENGTH = 4
turn  =0
class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7))
        self.piece = 1
        self.board_copy = self.board.copy()
        

    def print_board(self):
        print(np.flip(self.board, 0))

    def getValidLocation(self):
        valid_locations = []
        for col in range(numOfColumns):
            if self.board[numOfRows-1][col] == 0:
                valid_locations.append(col)
        return valid_locations

    def make_move(self, col):
        for row in range(6):
            if self.board[row][col] == 0:
                self.board[row][col] = self.piece
                break

    def winningMove(self , piecee): 
    #checking the rows for winning player
        for c in range(numOfColumns-3):
            for r in range(numOfRows):
                if(self.board[r][c] == piecee and self.board[r][c+1] == piecee and  self.board[r][c+2]== piecee and  self.board[r][c+3] == piecee):
                    return True

    #checking the columns for winning player
        for c in range(numOfColumns):
            for r in range(numOfRows-3):
                if(self.board[r][c] == piecee and self.board[r+1][c] == piecee and  self.board[r+2][c]== piecee and  self.board[r+3][c] == piecee):
                    return True

    #checking for positively sloped diagnols
        for c in range(numOfColumns-3):
            for r in range(numOfRows-3):
                if(self.board[r][c] == piecee and self.board[r+1][c+1] == piecee and  self.board[r+2][c+2]== piecee and  self.board[r+3][c+3] == piecee):
                    return True

    #checking for negatively sloped diagnols
        for c in range(numOfColumns-3):
            for r in range(3, numOfRows-3):
                if(self.board[r][c] == piecee and self.board[r-1][c+1] == piecee and  self.board[r-2][c+2]== piecee and  self.board[r-3][c+3] == piecee):
                    return True

    
    
    def drawBoard(self):
    # Draw the board squares
        for row in range(numOfRows):
            for col in range(numOfColumns):
                x = col * SQUARESIZE
                y = row * SQUARESIZE 
                pygame.draw.rect(screen, GREEN, (x, y + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.rect(screen, BLACK, (x, y+ SQUARESIZE, SQUARESIZE, SQUARESIZE), 4)

        # Draw the pieces
        for row in range(numOfRows):
            for col in range(numOfColumns):
                if self.board[row][col] == PLAYER_PIECE:
                    x = col * SQUARESIZE + SQUARESIZE/2
                    y = row * SQUARESIZE + SQUARESIZE/2
                    pygame.draw.circle(screen, RED, (int(x), HEIGHT-int(y)), RADIUS)
                elif self.board[row][col] == AGENT_PIECE:
                    x = col * SQUARESIZE + SQUARESIZE/2
                    y = row * SQUARESIZE + SQUARESIZE/2
                    pygame.draw.circle(screen, BLUE, (int(x), HEIGHT-int(y)), RADIUS)

        # Update the display
        pygame.display.update()


class DQNAgent:
    def __init__(self):
        self.state_size = (6, 7)
        self.action_size = 7
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=42, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.flatten().reshape(1,-1))
        return np.argmax(act_values[0])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1,-1))[0]))
            target_f = self.model.predict(state.reshape(1,-1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1,-1), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Main function
if __name__ == "__main__":
    env = ConnectFour()
    agent = DQNAgent()

    # Training
    for i in range(20):
        state = env.board
        done = False
        while not done:
            action = agent.act(state)
            legal_moves = env.getValidLocation()
            if action not in legal_moves and legal_moves:
                action = random.choice(legal_moves)
            env.piece = 1
            # row = env.get_next_open_row(action)
            # env.drop_piece(row , action, 1)
            env.make_move(action)
            
            next_state = env.board
            reward = 0
            
            if env.winningMove(1):
                reward = 1
                done = True
            elif len(legal_moves) == 0:
                done = True
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            print("it is training")

    # Save trained model
    agent.save("connect4-dqn.h5")
    
    # Play against the trained model
    agent.load("connect4-dqn.h5")
    # with GUI
    env.board = np.zeros((6, 7))
    env.piece = random.randint(1,2)
    game_over = False
    pygame.init()
    SQUARESIZE = 100
    WIDTH = numOfColumns * SQUARESIZE
    HEIGHT = (numOfRows+1) * SQUARESIZE
    RADIUS = int(SQUARESIZE/2 - 5)
    size = (WIDTH ,HEIGHT)
    screen = pygame.display.set_mode(size)
    env.drawBoard()
    pygame.display.update()
    font = pygame.font.SysFont("monospace",75)
    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK,(0, 0, WIDTH, SQUARESIZE))
                x_position = event.pos[0]
                if env.piece == 1:
                    pygame.draw.circle(screen , RED ,(x_position ,int(SQUARESIZE/2)), RADIUS )
            pygame.display.update()  

            if event.type == pygame.MOUSEBUTTONDOWN:
               
                if env.piece == 1:
                    x_position = event.pos[0]
                    col = int(math.floor(x_position/SQUARESIZE))

                    valid_locations = env.getValidLocation()
                    if col in valid_locations :
                        env.make_move(col)
               

                        if env.winningMove(PLAYER_PIECE):
                            label = font.render("Player 1 Wins", 1 ,RED)
                            screen.blit(label,(40,10))
                            game_over = True

                        env.piece = 3 - env.piece
                        env.print_board()
                        env.drawBoard()
                                  
        if env.piece == 2 and not game_over:
           
            env.print_board()
            env.print_board()
            action = agent.act(env.board)
            env.print_board()
         
            env.print_board()
            valid_locations = env.getValidLocation()
            if action in valid_locations :
                #wait before making the move
                pygame.time.wait(500)
               
                env.make_move(action)
                print("third")

                if env.winningMove(AGENT_PIECE):
                    label = font.render("Player 2 Wins", 1 ,BLUE)
                    screen.blit(label,(40,10))
                    game_over = True
                   

                env.piece = 3 - env.piece
                
                env.print_board()
                env.drawBoard()
                

        if game_over:
            pygame.time.delay(4000)

     