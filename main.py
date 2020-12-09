import numpy as np
from datetime import datetime

# Initialize environment
# Actions: 0 -> Left, 1 -> Up, 2 -> Right, 3 -> Down
# States:
# |  7  |  8  |  9  |  10 |
# |  4  |  Χ  |  5  |  6  |
# |  0  |  1  |  2  |  3  |
state_reward = -0.04

P = {
    0: {
        0: [(0.8, 0, state_reward, False), (0.1, 0, state_reward, False), (0.1, 4, state_reward, False)],
        1: [(0.8, 4, state_reward, False), (0.1, 0, state_reward, False), (0.1, 1, state_reward, False)],
        2: [(0.8, 1, state_reward, False), (0.1, 4, state_reward, False), (0.1, 0, state_reward, False)],
        3: [(0.8, 0, state_reward, False), (0.1, 1, state_reward, False), (0.1, 0, state_reward, False)]
    },
    1: {
        0: [(0.8, 0, state_reward, False), (0.1, 1, state_reward, False), (0.1, 1, state_reward, False)],
        1: [(0.8, 1, state_reward, False), (0.1, 0, state_reward, False), (0.1, 2, state_reward, False)],
        2: [(0.8, 2, state_reward, False), (0.1, 1, state_reward, False), (0.1, 1, state_reward, False)],
        3: [(0.8, 1, state_reward, False), (0.1, 2, state_reward, False), (0.1, 0, state_reward, False)]
    },
    2: {
        0: [(0.8, 1, state_reward, False), (0.1, 2, state_reward, False), (0.1, 5, state_reward, False)],
        1: [(0.8, 5, state_reward, False), (0.1, 1, state_reward, False), (0.1, 3, state_reward, False)],
        2: [(0.8, 3, state_reward, False), (0.1, 5, state_reward, False), (0.1, 2, state_reward, False)],
        3: [(0.8, 2, state_reward, False), (0.1, 3, state_reward, False), (0.1, 1, state_reward, False)]
    },
    3: {
        0: [(0.8, 2, state_reward, False), (0.1, 3, state_reward, False), (0.1, 6, -1.00, True)],
        1: [(0.8, 6, -1.00, True), (0.1, 2, state_reward, False), (0.1, 3, state_reward, False)],
        2: [(0.8, 3, state_reward, False), (0.1, 6, -1.00, True), (0.1, 3, state_reward, False)],
        3: [(0.8, 3, state_reward, False), (0.1, 3, state_reward, True), (0.1, 2, state_reward, False)]
    },
    4: {
        0: [(0.8, 4, state_reward, False), (0.1, 0, state_reward, False), (0.1, 7, state_reward, False)],
        1: [(0.8, 7, state_reward, False), (0.1, 4, state_reward, False), (0.1, 4, state_reward, False)],
        2: [(0.8, 4, state_reward, False), (0.1, 7, state_reward, False), (0.1, 0, state_reward, False)],
        3: [(0.8, 0, state_reward, False), (0.1, 4, state_reward, False), (0.1, 4, state_reward, False)]
    },
    5: {
        0: [(0.8, 5, state_reward, False), (0.1, 2, state_reward, False), (0.1, 9, state_reward, False)],
        1: [(0.8, 9, state_reward, False), (0.1, 5, state_reward, False), (0.1, 6, -1.00, True)],
        2: [(0.8, 6, -1.00, True), (0.1, 9, state_reward, False), (0.1, 2, state_reward, False)],
        3: [(0.8, 2, state_reward, False), (0.1, 6, -1.00, True), (0.1, 5, state_reward, False)]
    },
    6: {
        0: [(1.0, 6, -1.00, True)],
        1: [(1.0, 6, -1.00, True)],
        2: [(1.0, 6, -1.00, True)],
        3: [(1.0, 6, -1.00, True)]
    },
    7: {
        0: [(0.8, 7, state_reward, False), (0.1, 4, state_reward, False), (0.1, 7, state_reward, False)],
        1: [(0.8, 7, state_reward, False), (0.1, 7, state_reward, False), (0.1, 8, state_reward, False)],
        2: [(0.8, 8, state_reward, False), (0.1, 7, state_reward, False), (0.1, 4, state_reward, False)],
        3: [(0.8, 4, state_reward, False), (0.1, 8, state_reward, False), (0.1, 7, state_reward, False)]
    },
    8: {
        0: [(0.8, 7, state_reward, False), (0.1, 8, state_reward, False), (0.1, 8, state_reward, False)],
        1: [(0.8, 8, state_reward, False), (0.1, 7, state_reward, False), (0.1, 9, state_reward, False)],
        2: [(0.8, 9, state_reward, False), (0.1, 8, state_reward, False), (0.1, 8, state_reward, False)],
        3: [(0.8, 8, state_reward, False), (0.1, 9, state_reward, False), (0.1, 7, state_reward, False)]
    },
    9: {
        0: [(0.8, 8, state_reward, False), (0.1, 5, state_reward, False), (0.1, 9, state_reward, False)],
        1: [(0.8, 9, state_reward, False), (0.1, 8, state_reward, False), (0.1, 10, 1.00, True)],
        2: [(0.8, 10, 1.00, True), (0.1, 9, state_reward, False), (0.1, 5, state_reward, False)],
        3: [(0.8, 5, state_reward, False), (0.1, 10, 1.00, True), (0.1, 8, state_reward, False)]
    },
    10: {
        0: [(1.0, 6, 1.00, True)],
        1: [(1.0, 6, 1.00, True)],
        2: [(1.0, 6, 1.00, True)],
        3: [(1.0, 6, 1.00, True)]
    }
}

# create aliases for the actions
left, up, right, down = 0, 1, 2, 3


def value_iteration(P, gamma, theta=1e-10):
    # initialize state-value arbitrarily for every state s
    V = np.random.random(len(P))
    while True:
        max_delta = 0

        # initialize action-value function for all state-action pairs to 0
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        # iterate over every state available
        for s in range(len(P)):
            # cache the old value for the state s
            v = V[s]

            # check every possible transition from state s, to calculate state-action value function
            for a in range(len(P[s])):
                for prob, new_state, reward, done in P[s][a]:
                    if done:
                        # for final states (6, 10) future value is zero
                        value = reward
                    else:
                        value = reward + gamma * V[new_state]
                    # action-value function is calculated for every state-action pair
                    Q[s][a] += prob * value
            # state-value function is going to be the max action-value function from state s
            V[s] = np.max(Q[s])
            max_delta = max(max_delta, abs(v - V[s]))
        # check if the max changes for every state-value is greater than the threshold theta
        if max_delta < theta:
            break

    # optimal policy will be the action with the maximum action-value for every state s
    pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
    return V, pi


# This class represents a node
class Node:
    # Initialize the class
    def __init__(self, x: (), y: (), parent: ()):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # Sort nodes
    def __lt__(self, other):
        return self.f < other.f

    # Print node
    def __repr__(self):
        return '({0},{1},{2})'.format(self.x, self.y, self.f)

    def get_index(self):
        index = 4 * self.x + self.y
        if index >= 5:
            index += -1
        return index


def get_neighbours(node):
    neighbours = []
    for n in [(node.x - 1, node.y), (node.x + 1, node.y), (node.x, node.y - 1), (node.x, node.y + 1)]:
        index = 4 * n[0] + n[1]
        if 0 <= n[0] < 3 and 0 <= n[1] < 4 and index != 5:
            neighbours.append(Node(n[0], n[1], node))
    return neighbours


def a_star_search(initial_node: Node, goal_node: Node, v_star_function: []):
    # Create lists for open nodes and closed nodes
    open_list = []
    closed_list = []

    initial_node.h = v_star_function[initial_node.get_index()]
    initial_node.f = initial_node.h
    open_list.append(initial_node)

    # Loop until the open list is empty
    while len(open_list) > 0:

        # Sort the open list to get the node with the highest value first
        open_list.sort(reverse=True)

        # Get the node with the highest value
        current_node = open_list.pop(0)

        # Add the current node to the closed list
        closed_list.append(current_node)

        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != initial_node:
                path.append(current_node.get_index())
                current_node = current_node.parent
            # path.append(start)
            # Return reversed path
            path.append(initial_node.get_index())
            return path[::-1]

        # Loop neighbors
        for neighbor in get_neighbours(current_node):

            # Check if the neighbor is in the closed list
            for node in closed_list:
                if node == neighbor:
                    continue

            # Generate neighbor cost and values
            neighbor.h = v_star_function[neighbor.get_index()]
            neighbor.g = current_node.g + neighbor.h
            neighbor.f = neighbor.g + neighbor.h

            # Check if neighbor is in open list and if it has a lower f value
            if add_to_open(open_list, neighbor):
                open_list.append(neighbor)

    return None


# Check if a neighbor should be added to open list
def add_to_open(open_list, neighbor):
    for node in open_list:
        if neighbor == node and neighbor.f < node.f:
            return False
    return True


optimal_V = {}
for g in [0.9, 0.6, 0.2]:
    V_star_value, pi_star_value = value_iteration(P, g)
    optimal_V[g] = V_star_value

    print("For g: {:2f} the optimal V function is the following:".format(g))
    print(V_star_value)
    print("Optimal policy is the following:")
    policy = ""
    for i in [2, 1, 0]:
        for j in range(4):
            policy_index = 4 * i + j
            if policy_index == 5:
                policy += "| X "
            elif policy_index == 11:
                policy += "| O "
            elif policy_index == 7:
                policy += "| - "
            else:
                if policy_index > 5:
                    policy_index -= 1

                if pi_star_value[policy_index] == 0:
                    policy += "| ← "
                elif pi_star_value[policy_index] == 1:
                    policy += "| ↑ "
                elif pi_star_value[policy_index] == 2:
                    policy += "| → "
                else:
                    policy += "| ↓ "
        policy += "|\n"
    print(policy)

    a_star_path = a_star_search(Node(0, 0, None), Node(2, 3, None), V_star_value)
    print(a_star_path)
