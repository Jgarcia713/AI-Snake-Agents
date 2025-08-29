"""
File: runGame.py
Author: Jakob Garcia
Class: INFO 450
The controller to run the program. Currently uses the console. A sample of the output
looks like the image below, where S represents the head of the snake, T represents the tail,
O represents the body, [a] represents a fruit, [ ] represents an empty spot, and ===
represents a wall. There is output at the bottom as well for the coordinates of each
body part in the order they are connected to the snake.
=====================
=== O  O  O  T [ ]===
=== O [ ][ ][ ][ ]===
=== O  O  O [a][ ]===
===[ ][ ] O [ ][ ]===
===[ ][ ] O  S [ ]===
=====================
S(4, 5)O(3, 5)O(3, 4)O(3, 3)O(2, 3)O(1, 3)O(1, 2)O(1, 1)O(2, 1)O(3, 1)T(4, 1)

In the future, I would like to add more functionality for command line arguments
"""
import os
from snake import Grid, Directions
from snakeAgents import RandomAgent, AStarSearchAgent, QLearningAgent, ApproximateQAgent
import time


DELAY = 0.2 # constant for the delay between actions the Agents take.
GRID_SIZE = 7

def random_game(grid: Grid):
    """
    Run the game using a RandomAgent who randomly chooses moves
    Parameters: grid is an object of the Grid class
    """
    random_agent = RandomAgent(grid)
    valid = True

    while not random_agent.is_goal_state():
        os.system('cls')
        move = random_agent.get_successors()
        valid = grid.move_snake(move)

        print(grid)
        print(grid._snake)
        if not valid:
            break
        elif random_agent.is_goal_state(): 
            print("You Win!")
        time.sleep(DELAY)
    
    print("Nodes expanded:", random_agent.get_expanded_count())


def aStar_game(grid: Grid):
    """
    Run the game using a AStarSearchAgent who chooses moves based on paths created using the
    A* search algorithm
    Parameters: grid is an object of the Grid class
    """
    aStar_agent = AStarSearchAgent(grid)
    valid = True
    print(grid)

    move_count = 0
    while not aStar_agent.is_goal_state():
        moves = aStar_agent.aStar_search()
        for move in moves:
            os.system('cls')
            valid = grid.move_snake(move)
            move_count += 1
            print(grid)
            print(grid._snake)
            if aStar_agent.is_goal_state(): 
                break
            elif not valid:
                break
            time.sleep(DELAY)
        if not valid or aStar_agent.is_goal_state():
            break
        aStar_agent.update_fruit_coords()
    valid = True


def Qlearning_game(grid: Grid):
    os.system('cls')
    qLearning_agent = QLearningAgent(grid, gamma=0.7, alpha=0.3, num_training=30000, epsilon=1)
    # trials = qLearning_agent.deeper_training(10)
    qLearning_agent.simple_training()

    print(grid)
    input("Press enter to start")
    game_over = False
    move_count = 0
    state = qLearning_agent.get_snake_state()
    while not game_over:
        os.system('cls')
        move_count += 1
        action = qLearning_agent.compute_best_action(state)
        next_state, _, game_over = qLearning_agent.take_action(state, action)
        state = next_state

        print(grid)
        print(grid._snake)
        time.sleep(DELAY)


def approx_Qlearning_game(grid: Grid):
    os.system('cls')
    approx_learning_agent = ApproximateQAgent(grid, gamma=0.7, alpha=0.4, num_training=30000, epsilon=1)
    trials = approx_learning_agent.deeper_training(10)
    approx_learning_agent.simple_training()

    print(grid)
    input("Press enter to start")
    game_over = False
    move_count = 0
    state = approx_learning_agent.get_snake_state()
    while not game_over:
        os.system('cls')
        move_count += 1
        action = approx_learning_agent.compute_best_action(state)
        next_state, _, game_over = approx_learning_agent.take_action(state, action)
        state = next_state

        print(grid)
        print(grid._snake)
        time.sleep(DELAY)


def debug_play(grid: Grid):
    """
    Run the game without an agent, and instead take keyboard input to test the behavior of the game
    Parameters: grid is an object of the Grid class
    """
    move = input()
    valid = True
    while move != 'end':
        os.system('cls')
        if move.lower() in Directions.DIRECTIONS and move != '':
            valid = grid.move_snake(move.lower())

        print(grid)
        print(grid._snake)
        if valid:
            move = input()
        elif grid.has_won(): 
            print("You Win!")
            break
        else:
            break


def main():
    os.system('cls')

    grid = Grid(GRID_SIZE)
    print(grid)

    # debug_play(grid)
    # random_game(grid)
    # aStar_game(grid)
    # Qlearning_game(grid)
    approx_Qlearning_game(grid)


main()