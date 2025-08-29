"""
File: snakeAgents.py
Author: Jakob Garcia
Class: INFO 450
Description: This file is used to implement the logic for several different agents that 
will play the game snake, so that comparisons can be made between different algorithms.
"""
import random
from snake import Directions
from util import PriorityQueue
import numpy as np

def manhattan_distance(start_coords, target_coords) -> int:
    """
    Calculates the distance between two points in a grid-like path.
    Parameters:
        - start_coords: a starting coordinate pair
        - target_coords: an ending coordinate pair
    Returns: the sum of the absolute differences between the coordinates of the points.
    """
    return abs(start_coords[0] - target_coords[0]) + abs(start_coords[1] - target_coords[1])

class RandomAgent:
    """
    This class represents the random agent who will randomly choose where to move next, but
    will avoid moves that would end the game in the next state if it can help it. This means
    it will avoid running into walls or itself if there any other valid moves. 
    """
    def __init__(self, start_state):
        """
        Initialize the random agent with the starting state of the game
        Parameters: startingGameState is a Grid object 
        """
        self._game_state = start_state
        self._expanded = 0  # Bookkeeping number of search nodes expanded

    def is_goal_state(self) -> bool:
        """
        Returns whether this search state is a goal state of the problem. Aka, whether 
        the board is completely filled by the snake and no more fruits can be placed.
        """
        return self._game_state.has_won()

    def get_successors(self) -> str:
        """
        This function returns the action that will lead to a successor state. Since this agent acts randomly and we aren't 
        concerned with building a path, it simply evaluates all possible moves it can make, and randomly chooses
        one that won't result in an immediate game over. If there is no move that can avoid running into an obstacle
        choose a random direction and move there.
        Returns: a value of the Directions enum, indicating a direction to move in the form a char
        """
        successors = []
        body_coords = self._game_state.get_body_coords()
        curr_pos = body_coords[0]
        max_coord = self._game_state.get_size()+1

        for action in Directions.DIRECTIONS:
            next_pos = Directions.update_position(curr_pos, action)
            if max_coord < next_pos[0] or next_pos[0] < 0 or max_coord < next_pos[1] or next_pos[1] < 0:
                continue 
            elif tuple(next_pos) in self._game_state.wall_coords:
                continue                                    # ignore body collisions except for the tail
            elif tuple(next_pos) in body_coords[:-1] or (len(body_coords) == 2 and tuple(next_pos) == body_coords[-1]):
                continue
            successors.append(action)

        self._expanded += 1  
        if successors == []:  # If there were no options that avoided collisions, do a default action
            return random.choice(Directions.DIRECTIONS)
        return random.choice(successors) # return a random action
    
    def get_expanded_count(self) -> int:
        return self._expanded
    

class AStarSearchAgent:
    """
    This class represents an agent that uses the A* search algorithm in order to determine which actions would best result
    in fulfilling its goals. The agent primarily searches for the current fruit on the board, but if the fruit is unreachable from
    the snakes current position, it instead prioritizes finding the tail. If no moves can be made to reach the tail, a random action
    is then performed.
    """
    def __init__(self, game_state):
        self._game_state = game_state
        self._fruit_coords = game_state.get_fruit_coords()
        self._expanded = 0  # Bookkeeping number of search nodes expanded

    def get_start_state(self) -> tuple[int]:
        """
        Returns the start state; in this case the coordinates of the snake's head
        in the form a list of an x,y coord.
        """
        return self._game_state.get_body_coords()[0]

    def is_goal_state(self) -> bool:
        """
        Returns whether this search state is a goal state of the problem. Aka, whether 
        the board is completely filled by the snake and no more fruits can be placed.
        """
        return self._game_state.has_won()
    
    def is_subgoal_state(self, state) -> bool:
        """
        Returns whether this search state is a subgoal state of the problem. For this search
        agent, that is whether we have reached the current fruit on the board
        Parameters: state is tuple of x,y coordinates of where the snake head currently is.
        """
        return state == self._fruit_coords

    def update_fruit_coords(self):
        """
        Set the new coordinates of where the next fruit is.
        """
        self._fruit_coords = self._game_state.get_fruit_coords()

    def get_successors(self, curr_pos) -> list[tuple]:
        """
        Determine the successors of a given state. Is capable of determining which body parts cannot be outrun by the snake when moving
        towards the goal. It labels each body part with how many movements until the tail reaches that point, and if the distance between
        the current position and that body part is greater the number of movements until the tail takes that position, that path is valid.
        As we simulate the snake head moving, the number of moves for the tail to reach a point decreases for each move the snake makes
        Parameters: curr_pos is a tuple of coordinates indicating the state
        Returns: a list of valid successors represented as tuples of coordinate, action pairs
        """
        successors = []
        body_coords = self._game_state.get_body_coords()
        start_pos = self.get_start_state()
        snake_moves = manhattan_distance(start_pos, curr_pos)  # number of moves from the start position to the current one
        # Determine which body parts cannot be outrun by the snake when moving towards the goal. 
        obstacle_body_parts = [coord for tail_dist, coord in enumerate(body_coords[::-1]) 
                               if manhattan_distance(curr_pos, coord) <= tail_dist - snake_moves]
        max_coord = self._game_state.get_size()+1
        
        for action in Directions.DIRECTIONS:
            next_pos = tuple(Directions.update_position(curr_pos, action))
            # ignore moves that hit walls or are out of bounds. Also ignore if a move will run into a body part
            if max_coord < next_pos[0] or next_pos[0] < 0 or max_coord < next_pos[1] or next_pos[1] < 0:
                continue 
            if tuple(next_pos) in self._game_state.wall_coords:
                continue
            if tuple(next_pos) in obstacle_body_parts or (len(body_coords) == 2 and tuple(next_pos) == body_coords[-1]):
                continue
            successors.append( (next_pos, action) )

        self._expanded += 1  
        if successors == []:  # If there were no options that avoided collisions, do a default action
            action = random.choice(Directions.DIRECTIONS)
            next_pos = tuple(Directions.update_position(curr_pos, action))
            return [(next_pos, action)]
        
        return successors

    def get_cost_of_actions(self, actions: list) -> int:
        """
        Given a list of actions, return the cost associated with executing every action. 
        Parameters: actions is a list of strings (values of the Direction enum class)
        Returns: the cost associated with that particular set of actions
        """
        next_pos = self.get_start_state()
        start_pos = self.get_start_state()
        body_coords = self._game_state.get_body_coords()
        max_coord = self._game_state.get_size()+1

        cost = 0
        for action in actions:
            # Determine which body parts cannot be outrun by the snake when moving towards the goal
            obstacle_body_parts = [coord for tail_dist, coord in enumerate(body_coords[::-1]) 
                                   if manhattan_distance(next_pos, coord) <= tail_dist - manhattan_distance(next_pos, start_pos)]

            next_pos = Directions.update_position(next_pos, action) # update the x or y value depending on the action
            if max_coord < next_pos[0] or next_pos[0] < 0 or max_coord < next_pos[1] or next_pos[1] < 0:
                return 999999 
            if tuple(next_pos) in self._game_state.wall_coords: # walls or body collisions incur a high cost
                return 999999
            if tuple(next_pos) in obstacle_body_parts or (len(body_coords) == 2 and tuple(next_pos) == body_coords[-1]):
                return 999999

            cost += 1
        return cost

    def aStar_search(self):
        """
        Search the node that has the lowest combined cost and heuristic first in order to find fruits. If the fruit is determined to be 
        unreachable in the current starting state, the agent will instead search for the tail if available in order to hopefully open up
        a path to the fruit again
        Parameters: None
        Returns: a list of the actions to take to reach the specified goal (fruit or tail)        
        """
        heuristic = manhattan_distance # alias the heuristic and cost functions
        cost = self.get_cost_of_actions
        start = self.get_start_state()
        if self.is_subgoal_state(start): # if initial state is goal
            return []

        queue = PriorityQueue()
        # push successor info and a list to keep track of individual paths    
        queue.update((start, []), (cost([]) + heuristic(start, self._fruit_coords))) # cost + heuristic (as the priority)
        explored = set()
        while not queue.is_empty():
            state, path = queue.dequeue()

            if state in explored: # ignore states already explored
                continue
            elif self.is_subgoal_state(state): # whether a fruit has been reached
                return path

            explored.add(state)
            for next_state, action in self.get_successors(state):
                if next_state not in explored:
                    queue.update((next_state, path+[action]), (cost(path+[action]) + heuristic(next_state, self._fruit_coords)))

        # If the loop ended, there is no clear goal to the fruit, meaning the snake may be temporarily trapped. Search for the tail instead        
        # print("Initiating tail search")
        path = self._aStar_tail_search()
        # Go partially towards the tail and then reassess in the next iteration if the fruit is reachable
        return path[:len(path)//2 + 1] 

    def _aStar_tail_search(self):
        """
        Search the node that has the lowest combined cost and heuristic first in order to find the snakes tail.
        If the tail is unreachable, a random action is done.
        Parameters: None
        Returns: a list of the actions to take to reach the tail
        """
        heuristic = manhattan_distance # alias the heuristic and cost functions
        cost = self.get_cost_of_actions
        start = self.get_start_state()
        tail_coords = self._game_state.get_body_coords()[-1]
        if tail_coords == start: # if initial state is goal
            return []

        queue = PriorityQueue()
        # push successor info and a list to keep track of individual paths    
        queue.update((start, []), (cost([]) + heuristic(start, tail_coords))) # cost + heuristic (as the priority)
        explored = set()
        while not queue.is_empty():
            state, path = queue.dequeue()

            if state in explored: # ignore states already explored
                continue
            elif tail_coords == state: # tail was reached
                return path
                        
            explored.add(state)
            for next_state, action in self.get_successors(state):
                if next_state not in explored:
                    queue.update((next_state, path+[action]), (cost(path+[action]) + heuristic(next_state, tail_coords)))
        
        # print("Initiating random search")
        return random.choice(self.get_successors(start))[1] # tail was unreachable, this will return a random action

    def get_expanded_count(self) -> int:
        return self._expanded
    

class QLearningAgent:
    """
    This class represents an agent that uses Q-learning in order to determine which actions would best result
    in fulfilling its goals. It undergoes training episodes in order to learn, and then can be used to perform what
    it has learned.
    """
    def __init__(self, game_state, num_training=10000, epsilon=1, alpha=0.3, gamma=0.7):
        """
        Initialize the Q-learning agent. 
        Parameters:
            - alpha : the learning rate
            - epsilon: the exploration rate
            - gamma: discount factor
            - num_training: number of training episodes
        """
        self.num_training = num_training
        self.epsilon = epsilon
        self._epsilon_backup = self.epsilon
        self.alpha = alpha
        self.discount = gamma
        self._game_state = game_state
        self._last_fruit_coords = game_state.get_fruit_coords()
        self._expanded = 0  # Bookkeeping number of search nodes expanded
        self._q_table = {}
        self._loops = set() # keep track of if repetitive moves are being made
        self._move_count = 0
        self._loop_tolerance = self._game_state.get_size()

    def get_snake_state(self) -> tuple[int]:
        """
        Returns the start state; in this case the coordinates of the snake's body and the fruit coordinates
        """
        return (tuple(self._game_state.get_body_coords()), self._game_state.get_fruit_coords())
    
    def get_Qvalue(self, state, action) -> float:
        """
        Returns the Q value for a given state and action. Default value of 0 is returned
        if the state has not been encountered before
        """
        return self._q_table.get((state, action), 0)

    def compute_value(self, state) -> float:
        """
        Compute the maximum value of a state based on the Q values for each action
        Parameters: state is a collection of coordinates representing the positions the snake occupies, and the current fruit position
        Returns: the maximum value of the state
        """
        value = []
        for action in self.get_legal_actions(state[0][0]): 
          value.append(self.get_Qvalue(state, action))
        return max(value) if value != [] else 0

    def compute_best_action(self, state):
        """
        Compute the best action that can be done at a given state. The best action corresponds to the best Q value
        Parameter: state is a collection of coordinates representing the positions the snake occupies, and the current fruit position
        Returns: a string of the best action to take at the position
        """
        max_value = self.compute_value(state)
        actions = []
        for action in self.get_legal_actions(state[0][0]): 
            if self.get_Qvalue(state, action) == max_value:
                actions.append(action)
        return random.choice(actions) if actions != [] else random.choice(Directions.DIRECTIONS)

    def get_action(self, state):
        """
        Get the action that should be taken at the given state. Based on the epsilon parameter, a random move may be made
        to encourage exploration.
        Parameters: state is a collection of coordinates representing the positions the snake occupies, and the current fruit position
        Returns:  a string of the action to take at the position
        """
        if random.random() < self.epsilon: 
            action = random.choice(self.get_legal_actions(state[0][0])) 
        else:
            action = self.compute_best_action(state)
        return action
        
    def take_action(self, state, action):
        """
        A transition function that moves the snake and computes the rewards from moving from
        one state to the next. It also tracks whether the agent is moving in loops, and tries to break that movement
        Parameters: 
            - state: a collection of coordinates representing the positions the snake occupies, and the current fruit position
            - action: a string representing the action to be executed 
        Returns: a tuple of the next state, the reward from moving to the new state, and whether the end of the game has been reached
        """
        changed_move = False
        if self._move_count > self._game_state.get_size(): # Track if the agent is making repetitive moves in loops
            old_size = len(self._loops)
            self._loops.add((state, action))
            if len(self._loops) == old_size:
                self._loop_tolerance -= 1
            if self._loop_tolerance <= 0: # make a random move to hopefully exit the loop
                actions = [direc for direc in self.get_legal_actions(state[0][0]) if direc != action]
                if actions != []:
                    action = random.choice(actions)
                    self._loop_tolerance = self._game_state.get_size()
                    changed_move = True
            
        valid_move = self._game_state.move_snake(action)
        if not valid_move and not self._game_state.has_won(): # Penalize collisions 
            reward = -200  
            game_over = True
            return state, reward, game_over
        
        next_state = self.get_snake_state()
        if self._game_state.has_won():
            reward = 200
            game_over = True
            self._move_count = 0
            self._loops = set()
            self._loop_tolerance = self._game_state.get_size()
            return next_state, reward, game_over
        elif next_state[0][0] == self._last_fruit_coords: # reward eating a fruit
            reward = 100
            self._last_fruit_coords = self._game_state.get_fruit_coords()
            self._move_count = 0
            self._loops = set()
            self._loop_tolerance = self._game_state.get_size()
        else:
            if changed_move and (next_state, self.compute_best_action(next_state)) in self._loops:
                self._loop_tolerance = 0
            reward = -25  # cost of moving
            self._move_count += 1
        game_over = False  
        return next_state, reward, game_over

    def update(self, state, action, next_state, reward):
        """
        Update the value associated with a given state and action.
        Parameters:
            - state: a collection of coordinates representing the positions the snake occupies, and the current fruit position
            - action: a string representing the action to be executed 
            - next_state: the state after executing the action
            - reward: the reward associated with executing the action during the given state
        """
        q_val = self.get_Qvalue(state, action)
        self._q_table[(state, action)] = q_val + self.alpha*(reward + self.discount*self.compute_value(next_state) - q_val)

    def get_legal_actions(self, curr_pos) -> list[Directions]:
        """
        Determine the successors of a given state
        Parameters: curr_pos is a tuple of coordinates indicating the state
        Returns: a list of valid successors represented as tuples of coordinate, action pairs
        """
        successors = []
        body_coords = self._game_state.get_body_coords()
        max_coord = self._game_state.get_size()+1
        
        for action in Directions.DIRECTIONS:
            next_pos = tuple(Directions.update_position(curr_pos, action))
            # ignore moves that hit walls or are out of bounds. Also ignore if a move will run into a body part
            if max_coord < next_pos[0] or next_pos[0] < 0 or max_coord < next_pos[1] or next_pos[1] < 0:
                continue 
            if tuple(next_pos) in self._game_state.wall_coords:
                continue
            if tuple(next_pos) in body_coords[:-1] or (len(body_coords) == 2 and tuple(next_pos) == body_coords[-1]):
                continue
            successors.append(action)

        self._expanded += 1  
        if successors == []:  # If there were no options that avoided collisions, do a random action
            action = random.choice(Directions.DIRECTIONS)
            return [action]
        return successors
    
    def simple_training(self):
        """
        Train the agent over the number of training episodes specified in the constructor. An episode consists of a complete game
        of snake.
        """
        self.epsilon = self._epsilon_backup
        for i in range(self.num_training):
            if i % 200 == 0:
                print(i, self.epsilon)
            state = self.get_snake_state()
            game_over = False
            while not game_over:
                action = self.get_action(state)
                next_state, reward, game_over = self.take_action(state, action)
                self.update(state, action, next_state, reward)
                state = next_state
            self.epsilon = max(0.1, self.epsilon*0.9997)
            self._game_state.reset_game()

    def deeper_training(self, fruit_count=10):
        """
        Train the agent over a specific curriculum where the agent gets the chance to simulate playing at different lengths
        and has to eat fruit at each position on the grid a certain number of times
        """
        starting_states = {}
        starting_pos = None
        trials = 0
        size = self._game_state.get_size()
        self.epsilon = self._epsilon_backup
        food_counts = {}
        for i in range(1,size*size):
            food_counts[i] = {}
            for j in range(1, size+1):
                for k in range(1, size+1):
                    food_counts[i][(k,j)] = fruit_count

        for snake_len in range(1, size * size-size): # run for each snake size
            pos = self._last_fruit_coords
            while sum(food_counts[snake_len].values()) > 0: # ensure each position is visited
                state = self.get_snake_state()
                game_over = False
                trials += 1
                
                while not game_over: 
                    action = self.get_action(state)
                    next_state, reward, game_over = self.take_action(state, action)
                    # move the fruit to a location less visited
                    if not game_over and self._last_fruit_coords != pos and max(food_counts[self._game_state.get_snake_len()].values()) > 0:
                        valid_pos = [key for key, value in food_counts[self._game_state.get_snake_len()].items() 
                                     if value > 0 and key not in next_state[0]] # enforce that a fruit isn't placed where the body is.
                        if valid_pos != []:
                            pos = valid_pos.pop(random.randint(0, len(valid_pos)-1))
                            food_counts[self._game_state.get_snake_len()][pos] -= 1
                            self._game_state.move_fruit(pos)
                            if self._game_state.get_snake_len() not in starting_states: # store starting states of when fruit was reached
                                starting_states[self._game_state.get_snake_len()] = set()
                            starting_states[self._game_state.get_snake_len()].add(tuple(next_state[0]))
                            next_state = (next_state[0], pos)
                            self._last_fruit_coords = pos
                    self.update(state, action, next_state, reward)
                    state = next_state

                self.epsilon = max(0.2, self.epsilon*0.9999)
                self._game_state.reset_game()
                if self._last_fruit_coords != pos and max(food_counts[snake_len].values()) > 0: # set the next fruit on the board
                    valid_pos = [key for key, value in food_counts[snake_len].items() if value > 0]
                    pos = valid_pos.pop(random.randint(0, len(valid_pos)-1)) 
                    food_counts[snake_len][pos] -= 1
                    self._game_state.move_fruit(pos)
                    self._last_fruit_coords = pos
                elif self._last_fruit_coords == pos and max(food_counts[snake_len].values()) > 0: # snake died and didn't eat fruit
                    self._game_state.move_fruit(pos)
                    self._last_fruit_coords = pos

                if snake_len == 1: # place the snake somewhere compatible with the new fruit position
                    if self._game_state.get_body_coords()[0] == pos:
                        if pos == (1,1):
                            self._game_state.set_snake_pos(((1,2),))
                        else:
                            self._game_state.set_snake_pos(((1,1),))    
                else:
                    valid_pos = [state for state in starting_states[snake_len] if pos not in state]
                    if valid_pos == []: 
                        continue
                    starting_pos = random.choice(valid_pos)
                    self._game_state.set_snake_pos(starting_pos)   
            print(snake_len)
        self._game_state.reset_game()
        return trials


class ApproximateQAgent(QLearningAgent):
    """
    This class represents an agent that uses approximate Q-learning in order to determine which actions would best result
    in fulfilling its goals. It undergoes training episodes in order to learn, and then can be used to perform what
    it has learned.
    """
    def __init__(self, game_state, num_training=10000, epsilon=0.7, alpha=0.3, gamma=1):
        """
        Initialize the approximate Q-learning agent. Inherits methods from the Q-Learning agent class 
        Parameters:
            - alpha : the learning rate
            - epsilon: the exploration rate
            - gamma: discount factor
            - num_training: number of training episodes
        """
        super().__init__(game_state, num_training, epsilon, alpha, gamma)
        self._weights = {}

    def get_Qvalue(self, state, action):
        """
        Returns the approximate Q value for a given state and action. Default value of 0 is returned
        if the state has not been encountered before
        """
        features_vector = self.get_features(state, action)
        result = 0
        for feature in features_vector:
            weight = self._weights.get(feature, 0)
            result += features_vector[feature] * weight   
        return result

    def update(self, state, action, next_state, reward):
        """
        Update the feature weight associated with a given state and action.
        Parameters:
            - state: a collection of coordinates representing the positions the snake occupies, and the current fruit position
            - action: a string representing the action to be executed 
            - next_state: the state after executing the action
            - reward: the reward associated with executing the action during the given state
        """
        features_vector = self.get_features(state, action)
        difference = reward + self.discount*self.compute_value(next_state) - self.get_Qvalue(state, action)
        for feature in features_vector:
            weight = self._weights.get(feature, 0)
            self._weights[feature] = weight + self.alpha*difference*features_vector[feature]

    def get_features(self, state, action):
        """
        Defines the features used to approximate the Q-values for state action pairs
        Parameters:
            - state : a collection of coordinates representing the positions the snake occupies, and the current fruit position
            - action: a string representing the action to be executed 
        Returns: A dictionary mapping the features to their values
        """
        body_coords = state[0]
        fruit_coords = state[1]
        walls = self._game_state.wall_coords
        size = self._game_state.get_size()
        features = {}

        # features['head_x'] = body_coords[0][0]
        # features['head_y'] = body_coords[0][1]
        next_x, next_y = Directions.update_position(body_coords[0], action)
        # features['next_x'] = next_x
        # features['next_y'] = next_y

        if fruit_coords:
            dist_to_food = manhattan_distance((next_x, next_y), fruit_coords) # distance to the fruit
            features['food_dist'] = dist_to_food / 2 
            # features["inv_food_dist"] = size / (1 + dist_to_food)

            features["food_dx_after_move"] = (fruit_coords[0] - next_x)
            features["food_dy_after_move"] = (fruit_coords[1] - next_y)

            # dx = fruit_coords[0] - next_x
            # dy = fruit_coords[1] - next_y

            # features["food_north"] = 1 if dy < 0 else 0
            # features["food_south"] = 1 if dy > 0 else 0
            # features["food_west"] = 1 if dx < 0 else 0
            # features["food_east"] = 1 if dx > 0 else 0

             # import math

            # angle = math.atan2(dy, dx)  # gives angle in radians
            # features["angle_to_food_bucket"] = int(angle / (math.pi / 4)) # 8 directions (0-7)

            curr_dist = manhattan_distance(body_coords[0], fruit_coords)
            features["moving_closer_to_food"] = 1 if dist_to_food < curr_dist else -1

            # features["moving_toward_food_x"] = np.sign(fruit_coords[0] - next_x) 
            # features["moving_toward_food_y"] = np.sign(fruit_coords[1] - next_y) 
            # features["food_distance_change"] = (manhattan_distance(body_coords[0], fruit_coords) - dist_to_food) / 2 



        else:
            features['won'] = 1

        # dist_to_tail = manhattan_distance((next_x, next_y), body_coords[-2]) if len(body_coords) > 1 else 0
        # features["distance_to_tail"] = dist_to_tail

        # features['next_pos_safety'] = -1 if (next_x, next_y) in body_coords[:-1] or (next_x, next_y) in walls else 1
        # if features['next_pos_safety'] == 1:
        #     for direc in Directions.DIRECTIONS:
        #         next_pos = tuple(Directions.update_position((next_x, next_y), direc))
        #         if next_pos == body_coords[0]:
        #             continue
        #         elif next_pos in body_coords[:-1] or next_pos in walls:
        #             features[f'{direc}-safety'] = -1
        #         else:
        #             features[f'{direc}-safety'] = 1

        for key in features: # normalize the values based on the size of the grid
            features[key] = features[key] / size 
        
        return features
    