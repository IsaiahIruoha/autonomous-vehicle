import json
import time
from urllib import request
import heapq
import math

#FOR IN LAB:
#UNCOMMENT THIS

server_ip = "192.168.2.183"  # Replace with the actual server IP when available.
server = f"http://{server_ip}:5000"
authKey = "40"  # For the lab, this will be your team number.
team = 40

class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        # This allows the node to be compared based on cost in the priority queue.
        return self.cost < other.cost

def euclidean_distance(node1, node2):
    """
    Compute the Euclidean distance between two nodes.
    Used as the heuristic in the A* algorithm.
    """
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def reconstruct_path(node):
    """
    Given the goal node, backtrack to construct the path from start to goal.
    This returns the path as a list of (x, y) coordinates.
    """
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def a_star(start, goal, grid):
    """
    A* algorithm to compute the shortest path on a grid from start to goal.
    - start: Node representing the starting position.
    - goal: Node representing the destination.
    - grid: 2D list where 0 indicates free space and 1 indicates an obstacle.
    """
    rows, cols = len(grid), len(grid[0])
    # Priority queue for A* search (stores tuples of (priority, node)).
    pq = []
    heapq.heappush(pq, (0, start))
    # Set to track visited positions.
    visited = set()

    while pq:
        # Pop the node with the lowest total estimated cost.
        _, current = heapq.heappop(pq)

        # Check if we have reached the goal.
        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current)

        # Mark current node as visited.
        visited.add((current.x, current.y))

        # Consider the four possible moves (up, down, left, right).
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = current.x + dx, current.y + dy
            # Ensure we remain within the grid bounds and that the cell is not an obstacle.
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited and grid[new_x][new_y] == 0:
                neighbor = Node(new_x, new_y, current.cost + 1, current)
                # f(n) = g(n) + h(n) where g(n) is the current path cost and h(n) is the heuristic.
                total_cost = neighbor.cost + euclidean_distance(neighbor, goal)
                heapq.heappush(pq, (total_cost, neighbor))
    # If no path is found, return None.
    return None

# --------------------------
# VPFS API Functions
# --------------------------
# These functions abstract the REST API calls needed to interact with the VPFS server.

def get_fares():
    """
    Retrieve the list of available fares from the VPFS server.
    """
    res = request.urlopen(server + "/fares")
    if res.status == 200:
        # The response is expected in JSON format.
        return json.loads(res.read())
    else:
        print("Error fetching fares")
        return []

def claim_fare(fare_id):
    """
    Attempt to claim a fare using its ID.
    Returns True if successful, otherwise False.
    """
    res = request.urlopen(server + f"/fares/claim/{fare_id}?auth={authKey}")
    if res.status == 200:
        response = json.loads(res.read())
        return response['success']
    return False

def get_position():
    """
    Get the current vehicle position from the VPFS GPS API.
    Returns a Node representing the vehicle's position.
    """
    res = request.urlopen(server + f"/WhereAmI/{team}")
    if res.status == 200:
        # The position data is expected to be in a list with one dictionary element.
        position_data = json.loads(res.read())
        position = position_data[0]['position']
        return Node(position['x'], position['y'])
    return None

# --------------------------
# Main Operation Logic
# --------------------------
# The following block simulates the operations of an autonomous taxi:
# 1. Retrieve available fares.
# 2. Attempt to claim the first unclaimed fare.
# 3. Get the current vehicle position.
# 4. Use A* path planning to compute a path to the fare's drop-off destination.
if __name__ == "__main__":
    # Step 1: Retrieve the list of fares.
    fares = get_fares()
    # Iterate through the list to find an unclaimed fare.
    for fare in fares:
        if not fare['claimed']:
            # Step 2: Attempt to claim the fare.
            if claim_fare(fare['id']):
                print(f"Claimed fare {fare['id']}")
                # Step 3: Get the current vehicle position.
                start = get_position()
                if start is None:
                    print("Could not retrieve vehicle position.")
                    break
                # The destination is provided by the VPFS fare API in map coordinates.
                goal = Node(fare['dest']['x'], fare['dest']['y'])
                
                # Step 4: Define a mock grid.
                # For real implementation, the grid should represent the map with obstacles.
                # Here we use a simple 100x100 grid with all cells free (0 means free).
                grid = [[0 for _ in range(100)] for _ in range(100)]
                
                # Compute the path using A* algorithm.
                path = a_star(start, goal, grid)
                if path:
                    print("Path to destination:", path)
                else:
                    print("No path found to the destination.")
                break  # Exit after processing one fare.
            else:
                print(f"Failed to claim fare {fare['id']}")
        else:
            print(f"Fare {fare['id']} already claimed")
