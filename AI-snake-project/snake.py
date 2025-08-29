"""
File: snake.py
Author: Jakob Garcia
Class: INFO 450
Description: This file implements the logic for running a game of snake. There are several classes
that make up this file. The bulk of the logic for running the game happens in the Grid class, which
represents the state of the snake world at a given point in time. The snake itself is implemented as
a doubly linked structure that is imposed onto the grid. There is also an enum class for the directions/
actions the snake can take. The game follows the standard rules of snake, but more details about the rules/
allowed actions can be read in the Grid class's move_snake() method.
"""
import random

class Directions:
    """
    An enum class for the different directions one can head within the grid.
    It also contains a static method for translating a given coordinate pair to
    a new position based on a direction to head.
    """
    NORTH = 'w' 
    WEST = 'a' 
    SOUTH = 's' 
    EAST = 'd'                               # These are mapped to the wasd keys
    DIRECTIONS = [NORTH, WEST, SOUTH, EAST]  # A list of the values for checking if an inputted value is valid
    
    @staticmethod
    def update_position(pos: list[int], direction: str) -> list[int]:
        """
        This static method is used to generate the next coordinate pair if we moved in a specified direction
        from a given coordinate position.
        Parameters:
            - pos : a list of an x,y coordinate pair
            - direction : a specified direction to move (this is typically a value of this class)
        Returns: a list which is a pair of updated x,y coords 
        """
        movements = {'w': [pos[0], pos[1] - 1],
                     'a': [pos[0] - 1, pos[1]],
                     's': [pos[0], pos[1] + 1],
                     'd': [pos[0] + 1, pos[1]]}
        return movements[direction]


class Grid:
    """
    This class represents the grid or world the game takes place in. The grid 
    is made up of GridPos objects. The class contains information about the main
    Snake in the grid, and the coordinates of the Snake as well.
    """
    def __init__(self, size: int) -> None:
        """
        Initialize the grid using a given size, to create a size x size grid.
        Fill each position with a GridPos object.
        Parameters: size is an int used for the size of the grid
        """
        assert size >= 3, "ERROR: the size must be >= 3"
        self.wall_coords = []
        for i in range(0, size+2):
            self.wall_coords.append((i,0))
            self.wall_coords.append((i,size+1))
        for i in range(0, size+2):
            self.wall_coords.append((0,i))
            self.wall_coords.append((size+1,i))
        
        x, y = random.randint(1, size), random.randint(1, size)
        self._snake = Snake((x,y)) # Initialize the Snake randomly on the board  
        self._size = size # size of the grid
        self._fruit_coords = None
        self.generate_fruit()  # Randomly add a fruit to the grid
        self._won = False      # a flag for if the game was won (no fruit can be placed)

    def move_snake(self, direction: str) -> bool:
        """
        Move the snake on the board and shift the entire body to where the previous
        body part was. If a fruit was consumed on this move, grow the size of the snake.
        If the snake runs into a body part or wall, end the game. The snake is able to move into
        where its tail is currently at, as the tail will move when the head moves. While the
        snake is able to run into its own body, it is not able to move backwards into itself, 
        meaning that moving north and then trying to move south is not considered a valid action,
        and nothing would happen for that input. Implementation wise, the snake moves by creating 
        a new head at the position the snake is moving to, and adding the rest of the body to that 
        head, and then removing the tail/last part of the snake. When a fruit is consumed, the tail 
        is just not removed.
        Parameters: A char in "wasd" indicating a direction to move
        Returns: True if the move was valid, and False otherwise (False will end the game)  
        """
        snake_coords = self._snake.get_coords()
        # Get the coordinates of the next position of the snake. A copy of the coords list is used here so
        # that backwards movements get stalled and don't update the actual position.
        coords =  Directions.update_position(snake_coords[0], direction)
        new_fruit = False            # flag for if we need to add a new fruit to the grid
        new_pos = (coords[0], coords[1]) # the new position the snake is moving to
        
        # Handle collisions
        if new_pos in self.wall_coords:
            # print("Hit wall")
            return False      # Prevent moving backwards into yourself and ending the game (Can only move forward or turn)
        elif new_pos in snake_coords and new_pos == snake_coords[1]:
            return True       # edge case for allowing movement into the tail, since the tail will move
        elif new_pos in snake_coords and new_pos != snake_coords[-1]: 
            # print("Hit body")
            return False
        elif new_pos == self._fruit_coords: 
            new_fruit = True

        # move the snake
        if not new_fruit: # move snake with a body when no fruit is eaten
            self._snake.update_snake(new_pos, False)
        else: # Add a new fruit to the board when one is consumed
            self._snake.update_snake(new_pos, True)
            valid = self.generate_fruit()
            if not valid: # If no fruit was able to be generated, the board is filled, and the game ends
                self._won = True
                return False
        return True
    
    def generate_fruit(self) -> bool:
        """
        Randomly add a fruit to one of the remaining unoccupied spaces. If all spots are occupied
        the game must be over.
        Returns: False if no fruits were added and True otherwise
        """
        if self._size * self._size == self._snake.length: # if the area of the grid == area of the snake
            self._fruit_coords = None
            return False
        # gather all positions that are not walls or occupied by the snake and randomly choose one to add fruit to
        available_positions = [(col, row) for row in range(1,self._size+1) for col in range(1,self._size+1) 
                               if ((col, row) not in self._snake.get_coords() and (col, row) not in self.wall_coords)]
        self._fruit_coords = random.choice(available_positions) 
        return True

    def set_snake_pos(self, coords):
        """
        Update the coordinate positions of the snake
        """
        self._snake.set_coords(coords)
    
    def move_fruit(self, coords):
        """
        Set the coordinates of the fruit to the parameter coordinates
        """
        self._fruit_coords = coords
    
    def has_won(self) -> bool:
        return self._won
    
    def get_body_coords(self) -> list[tuple]:
        """
        Return a list of the coordinates of where every body part of the snake is
        """
        return self._snake.get_coords()
    
    def get_fruit_coords(self) -> tuple[int]:
        """
        Returns the tuple of coordinates of where the current fruit is
        """
        return self._fruit_coords
    
    def get_size(self) -> int:
        return self._size
    
    def get_snake_len(self) -> int:
        return self._snake.length
    
    def reset_game(self):
        """
        Reset the grid and all objects necessary to play a game of snake. This just calls the constructor again 
        so a new grid object does not need to be made.
        """
        self.__init__(self._size)

    def __str__(self) -> str: 
        grid = ""
        for row in range(self._size+2):
            for col in range(self._size+2):
                if (col, row) in self.wall_coords:
                    grid += '==='
                elif (col, row) == self._snake.get_coords()[0]:
                    grid += ' S '
                elif (col, row) == self._snake.get_coords()[-1]:
                    grid += ' T '
                elif (col, row) in self._snake.get_coords():
                    grid += ' O '
                elif (col, row) == self._fruit_coords:
                    grid += '[a]'
                else:
                    grid += '[ ]'
            grid += '\n'
        return grid.strip()
    

class Snake:
    """
    This class represents the snake that exists on the board. This class is used to
    keep track of the head of the snake, which is connected with the rest of the body.
    Implementation wise, the snake exists as a doubly linked list with quick access to the
    head and tail nodes. It also keeps track of how many nodes/SnakeBodys are in the structure. 
    """
    def __init__(self, grid_coords: tuple[int]) -> None:
        """
        Initialize the head and tail to be a SnakeBody at a given GridPos. Initial length is 1.
        Parameters: grid_location is a GridPos object the snake is going to be stored at.
        """
        self._snake_coords = [grid_coords]
        self.length = 1

    def update_snake(self, grid_coords: tuple[int], grow_snake: bool) -> None:
        """
        Update the coordinates that the snake occupies. If the snake ate a fruit, the tail coordinate
        is not lost.
        Parameters:
            - grid_coords: a coordinate pair tuple representing a position on a grid
            - grow_snake: a boolean indicating whether the snake should increase in length
        """
        if not grow_snake:
            self._snake_coords = self._snake_coords[:-1]
        else:
            self.length += 1
        self._snake_coords = [grid_coords] + self._snake_coords

    def get_coords(self):
        """
        Returns a list of tuples of the coordinates of the body parts of the snake
        """
        return self._snake_coords
    
    def set_coords(self, coords):
        """
        Set the coordinates of the snake to a given group of tuples
        """
        self._snake_coords = list(coords)
        self.length = len(self._snake_coords)

    def __str__(self) -> str:
        snake = ""
        for i in range(len(self._snake_coords)):
            if i == 0:
                snake += "S"
            else:
                snake += "O"
            snake += f"{self._snake_coords[i]}"
        return snake