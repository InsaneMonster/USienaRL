# Import packages

import numpy


class SumTree(object):
    """
    Sum tree (a version of a binary tree in which all the child nodes sum up in the parent node) system. It can store
    any kind of data until _capacity is reached.

    Attributes:
        - _capacity: the amount of data of any kind that can be stored inside the tree
    """

    def __init__(self,
                 capacity: int):
        # Define the tree _capacity: the number of leaf nodes that contains experiences
        self.capacity: int = capacity
        # Generate the tree with all nodes values = 0
        # Note: the number of nodes in a binary tree is equal to the sum of:
        # - Leaf nodes: _capacity
        # - Parent nodes: _capacity - 1 if _capacity is even, _capacity if _capacity is odd
        if self.capacity % 2 == 0:
            self._tree: [] = numpy.zeros(2 * self.capacity - 1)
        else:
            self._tree: [] = numpy.zeros(2 * self.capacity)
        # Initialize the data as a list of objects as big as the tree nodes and relative pointer
        self._data: [] = numpy.zeros(capacity, dtype=object)
        self._data_pointer: int = 0

    def add(self,
            data, priority_value: float):
        """
        Add a set of data (e.g. a sample) to the sum tree, with the given priority value.

        It also updates the sum tree with the new experiences in order to propagate the sum of priorities to the root.

        :param data: the data to insert (e.g. a sample)
        :param priority_value: the priority value associated with such inserted data
        """
        # Compute the tree index in which to put the experience with the given priority
        # Note: the tree is filled left to right
        tree_index: int = self._data_pointer + self.capacity - 1
        # Update the leaf at the computed tree index
        self.update(tree_index, priority_value)
        # Update data frame at the current pointer and increment it
        self._data[self._data_pointer] = data
        self._data_pointer += 1
        # If next data pointer is above _capacity, reset it to the beginning (it is used to overwrite)
        if self._data_pointer >= self.capacity:
            self._data_pointer = 0

    def get(self,
            priority_value: float) -> int:
        """
        Get the node index which stores the given priority value if found or with the last bottom node otherwise.

        It searches for the node using a binary search algorithm from left to right.

        :param priority_value: the priority value to search
        :return: the index of the found node or the index of the last bottom left node
        """
        # Search for the given priority value until the bottom of the tree is found
        parent_index: int = 0
        while True:
            # Compute left and right child indexes of current parent index
            child_index_left: int = 2 * parent_index + 1
            child_index_right: int = 2 * parent_index + 2
            # If bottom is reached, the search ends with last result
            if child_index_left >= len(self._tree):
                leaf_index: int = parent_index
                break
            # Search for the highest priority node with respect to with the given priority value
            else:
                if priority_value <= self._tree[child_index_left]:
                    parent_index = child_index_left
                else:
                    priority_value -= self._tree[child_index_left]
                    parent_index = child_index_right
        # Return the found leaf index of the tree
        return leaf_index

    def update(self,
               tree_index: int, priority_value: float):
        """
        Update all the tree up to the root starting from a change of priority at the given index with the given priority value.

        :param tree_index: the index of the node in which to change the priority value
        :param priority_value: the new priority value to set in such node
        """
        # Compute the score value update at the given index by difference
        # Note: this is used to update the sum of priorities assigned to parent nodes in the tree
        score_value_update: float = priority_value - self._tree[tree_index]
        # Update the priority at the given index
        self._tree[tree_index] = priority_value
        # Propagate through the tree with respect with the sum tree definition
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self._tree[tree_index] += score_value_update

    def get_priority(self,
                     tree_index: int):
        """
        Get the priority stored at the given index (in the node at the given index) if found, None otherwise.

        :param tree_index: the node index to get the priority from
        :return: the priority value found at the given tree index, None if not found
        """
        # Get the node of the tree at the given index
        if tree_index >= len(self._tree):
            return None
        return self._tree[tree_index]

    def get_data(self,
                 tree_index: int):
        """
        Get the data stored at the given index if found, None otherwise.

        :param tree_index: the node index to get the associated data from
        :return: the data found at the given tree index, None if not found
        """
        if tree_index >= len(self._tree):
            return None
        # Compute the data index relative to the given tree index
        data_index: int = tree_index - self.capacity + 1
        # Return the data stored at that index
        return self._data[data_index]

    @property
    def leafs(self):
        """
        Get the leaf nodes of the tree.

        :return: the last elements in the tree up to _capacity
        """
        # Get the leaf nodes on the tree
        return self._tree[-self.capacity:]

    @property
    def total_priority(self):
        """
        Get the root node of the tree (which stores the sum of priorities on all the child nodes recursively).

        :return: the root of the tree
        """
        # Get the root node of the tree
        return self._tree[0]

    @property
    def size(self):
        """
        Get the size of the sum-tree with respect to contained data.

        :return: the size of the contained data in the sum-tree.
        """
        return self._data.size
