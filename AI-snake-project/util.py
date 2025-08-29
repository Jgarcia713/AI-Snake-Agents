"""
File: util.py
Author: Jakob Garcia
Class: INFO 450
Description: A file full of extra classes/data structures necessary to run all algorithms
"""

class QueueNode:
    """
    Simple class for storing the value and priority of an item in a priority queue
    """
    def __init__(self, value, priority):
        """
        Parameters: 
            - value is the item to be stored in the priority queue
            - priority is an integer representing the priority of the item.
        """
        self._item = value
        self._priority = priority

    def get_item(self):
        return self._item
    
    def get_priority(self):
        return self._priority
    
    def set_priority(self, value):
        self._priority = value

    def __eq__(self, other):
        """
        Defines the equality operator between QueueNodes
        Parameters: other is expected to be an object of the QueueNode class as well
        Returns: True so long as the attributes of both objects are the same
        """
        return self._item == other._item and self._priority == other._priority

    def __lt__(self, other):
        """
        Define the comparison operators between QueueNodes
        Parameters: other is expected to be an object of the QueueNode class as well
        Returns: True so long as the priority of this node is less than the other node priority
        """
        return self._priority < other._priority

class PriorityQueue:
    '''
    This class represents a standard priority queue that uses a min heap implementation to quickly access the current minimum value
    '''
    def __init__(self):
        '''
        Initialize the heap
        '''
        self._array = []

    def _sink(self, i):
        """
        The sink method for a heap. Sinks down a value if the value is larger than its children, by swapping the two
        values.
        Parameters: i is the index of the value we are looking to sink
        Returns: None
        """
        value = self._array[i]
        min_child_index = -1
        min_child = value
        start = i * 2 + 1
        end = min(i * 2 + 2, len(self._array)-1)
        for k in range(start, end+1):
            if self._array[k] < min_child:
                min_child = self._array[k]
                min_child_index = k
        
        if min_child_index != -1:
            self._array[min_child_index] = value
            self._array[i] = min_child
            self._sink(min_child_index)

    def _swim(self, start):
        """
        The swim method for a heap. Swims up a value if that value is smaller than its parent by swapping the two values.
        Parameters: start is the index of the value we are looking to swim up.
        Returns: None
        """
        if 0 >= start:
            return
        
        value = self._array[start]
        parent_index = max(((start-1)//2), 0)

        if value < self._array[parent_index]:
            self._array[start] = self._array[parent_index]
            self._array[parent_index] = value
            self._swim(parent_index)

    def enqueue(self, item, priority):
        """
        Add an item to the priority queue. It's position in the queue depends on its priority.
        It is first added to the end of the heap and them swims up.
        Parameters: item is the value to add to the queue
        Returns: None
        """
        node = QueueNode(item, priority)
        self._array.append(node)
        self._swim(len(self._array)-1)

    
    def dequeue(self):
        """
        Remove the item at the head of the queue--since this priority queue uses a min heap, that is the minimum value in the array. 
        The last item of the heap is set to the head, and then is sunk down to maintain heap ordering. Throws an exception if the queue is empty.
        Parameters: None
        Returns: the value at the head of the queue (minimum element).
        """
        if self.is_empty():
            raise Exception("EmptyQueueException")
        elif len(self._array) == 1:
            return self._array.pop().get_item()
        value = self._array[0]
        self._array[0] = self._array.pop()
        self._sink(0)
        return value.get_item()

    def is_empty(self):
        """
        Returns whether the queue is empty
        """
        return self._array == []
    
    def update(self, item, priority):
        """
        Update the priority of an item in the priority queue. This may change the items position in the queue. If the
        item does not exist in the queue, add it instead.
        Parameters:
            - item : the item to update
            - priority : the new priority that item may be set with
        Returns: True if the item was updated, False otherwise
        """
        for i in range(len(self._array)):
            if self._array[i].get_item() == item:
                if self._array[i].get_priority() <= priority:
                    return False
                else:
                    self._array[i].set_priority(priority)
                    self._swim(i)
                    return True
                
        self.enqueue(item, priority)
        return True             

    def __str__(self):
        result = ""
        for item in self._array:
            result += f"{item.get_item()} "
        return result
