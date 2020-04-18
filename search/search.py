# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # Getting the start Position of Pacman using problem.getStartState function
    startPositionOfPacman = problem.getStartState()
    # A List to keep track of visited nodes
    visited = []
    # Using Stack data structure to insert elements and pop elements in the fringe
    fringe = util.Stack()
    # path: returns the route to the Goal
    path = []
    # First Push the pacman position in stack
    fringe.push((startPositionOfPacman, [], 0))
    # If pacman position itself is the goal return
    if(problem.isGoalState(startPositionOfPacman)):
        return path
    # Otherwise iterate till the stack is empty
    while (fringe.isEmpty() == False):
        # Get the last node,action and cost from stack
        current_node, current_action, current_cost = fringe.pop()
        path = current_action
        # If the node is goal state the break out of the loop and return path
        if problem.isGoalState(current_node):
            break
        # If the node is not visited then append it to the stack
        visited.append(current_node)
        # If the node is not goal state then iterate through the succesors of the current node
        successorsOfCurrentNode = problem.getSuccessors(current_node)
        for successor_node, successor_action, successor_cost in successorsOfCurrentNode:
            # Check if the node is already visited
            if successor_node in visited:
                # If the node is already visited then continue and do nothing
                continue
            else:
                # If the node is not visited then calculate the path,cost and action
                entirepath = path + [successor_action]
                cost = problem.getCostOfActions(entirepath)
                # Now push the successor node, successor node path and cost into stack
                fringe.push((successor_node, entirepath, cost))
    return path  # Return the final path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Getting the start Position of Pacman using problem.getStartState function
    startPositionOfPacman = problem.getStartState()
    # A List to keep track of visited nodes
    visited = []
    # Using Queue data structure to insert elements and pop elements in the fringe
    fringe = util.Queue()
    # path: returns the route to the Goal
    path = []
    # First Push the pacman position in Queue
    fringe.push((startPositionOfPacman, [], 0))
    # If pacman position itself is the goal return
    if(problem.isGoalState(startPositionOfPacman)):
        return path
    # Otherwise iterate till the Queue is empty
    while (fringe.isEmpty() == False):
        # Get the first node,action and cost inserted from Queue
        current_node, current_action, current_cost = fringe.pop()
        path = current_action
        # If the node is goal state thne break out of the loop and return path
        if problem.isGoalState(current_node):
            break
        # If the current node is visited then continue running the loop and do nothing
        if current_node in visited:
            continue
        else:
            # If the not visited then add it to the list of visited nodes and loop through the successors of the current node
            visited.append(current_node)
            successorsOfCurrentNode = problem.getSuccessors(current_node)
            for successor_node, successor_action, successor_cost in successorsOfCurrentNode:
                action = path + [successor_action]
                cost = problem.getCostOfActions(action)
                # Now push the successor node, successor node path and cost into stack
                fringe.push((successor_node, action, cost))
    return path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Getting the start Position of Pacman using problem.getStartState function
    startPositionOfPacman = problem.getStartState()
    # A List to keep track of visited nodes
    visited = []
    # Using Queue data structure to insert elements and pop elements in the fringe
    fringe = util.PriorityQueue()
    # path: returns the route to the Goal
    path = []
    # priority of the queue (initialize it to 0 at first)
    priority = 0
    # First Push the pacman position in Queue
    fringe.push((startPositionOfPacman, [], 0), priority)
    # If pacman position itself is the goal return
    if(problem.isGoalState(startPositionOfPacman)):
        return path
    # Otherwise iterate till the PriorityQueue is empty
    while (fringe.isEmpty() == False):
        # Pop the node values from PriorityQueue which has least cost function
        currentnodevalues = fringe.pop()
        path = currentnodevalues[1]
        # If the node is the goal node then return the path
        if problem.isGoalState(currentnodevalues[0]):
            break
            # If the node is not yet visited
        if currentnodevalues[0] in visited:
            continue
        else:
            # If the not visited then add it to the list of visited nodes and loop through the successors of the current node
            visited.append(currentnodevalues[0])
            successorsOfCurrentNode = problem.getSuccessors(currentnodevalues[0])
            for successor_node, successor_action, successor_cost in successorsOfCurrentNode:
                action = path + [successor_action]
                priority = problem.getCostOfActions(action)
                # Update the priority of the nodes according to the heuristic cost it takes to reach the goal and insert them in PriorityQueue
                fringe.update((successor_node, action, priority), priority)
    return path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    import searchAgents
    # Getting the start Position of Pacman using problem.getStartState function
    startPositionOfPacman = problem.getStartState()
    # A List to keep track of visited nodes
    visited = []
    # Using Queue data structure to insert elements and pop elements in the fringe
    fringe = util.PriorityQueue()
    heuristic_cost = heuristic(startPositionOfPacman, problem) + 0
    # path: returns the route to the Goal
    path = []
    # priority of the queue (initialize it to 0 at first)
    priority = 0
    # First Push the pacman position in Queue
    fringe.push((startPositionOfPacman, [], 0), heuristic_cost)
    # If pacman position itself is the goal return
    if(problem.isGoalState(startPositionOfPacman)):
        return path
    # Otherwise iterate till the PriorityQueue is empty
    while (fringe.isEmpty() == False):
        # Pop the node values from priority which has least cost function
        currentnodevalues = fringe.pop()
        path = currentnodevalues[1]
        # If the node is the goal node then return the path
        if problem.isGoalState(currentnodevalues[0]):
            break
        # If the node is not yet visited
        if currentnodevalues[0] in visited:
            continue
        else:
            # If the not visited then add it to the list of visited nodes and loop through the successors of the current node
            visited.append(currentnodevalues[0])
            successorsOfCurrentNode = problem.getSuccessors(currentnodevalues[0])
            for successor_node, successor_action, successor_cost in successorsOfCurrentNode:
                action = path + [successor_action]
                cost = problem.getCostOfActions(action)
                heuristic_cost = cost + heuristic(successor_node, problem)
                # Update the priority of the nodes according to theheuristic  cost it takes to reach the goal and insert them in PriorityQueue
                fringe.update((successor_node, action, cost), heuristic_cost)
    return path  # Return the path
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
