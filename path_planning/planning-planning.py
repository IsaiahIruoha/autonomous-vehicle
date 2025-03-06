import json
import time
from urllib import request
import heapq
import math

# VPFS Server Configuration
# For in-lab testing, use the provided server details.
server_ip = "192.168.2.183"  # Replace with the actual server IP when available.
server = f"http://{server_ip}:5000"
authKey = "40"  # For lab testing, this is your team number.
team = 40

# The following A* implementation calculates a path from a start to a goal position on a grid.
# In a real application, the grid should reflect the actual map (obstacles, roads, etc.).
class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        # Allow comparison based on cost for the priority queue.
        return self.cost < other.cost

def euclidean_distance(node1, node2):
    """Heuristic: Euclidean distance between two nodes."""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def reconstruct_path(node):
    """Backtrack from goal to start to build the path."""
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def a_star(start, goal, grid):
    """
    A* algorithm to compute the shortest path from start to goal on a grid.
    - start: starting Node.
    - goal: destination Node.
    - grid: 2D list representing the map (0 = free, 1 = obstacle).
    """
    rows, cols = len(grid), len(grid[0])
    pq = []
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)

        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current)

        visited.add((current.x, current.y))

        # Explore neighbors: up, down, left, right.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = current.x + dx, current.y + dy
            if 0 <= new_x < rows and 0 <= new_y < cols:
                if (new_x, new_y) not in visited and grid[new_x][new_y] == 0:
                    neighbor = Node(new_x, new_y, current.cost + 1, current)
                    total_cost = neighbor.cost + euclidean_distance(neighbor, goal)
                    heapq.heappush(pq, (total_cost, neighbor))
    return None

# The VPFS server provides the following endpoints:
#
# 1. /match?auth=<auth>
#    Returns the current match status:
#    {
#      "mode": "home" | "lab" | "match",
#      "match": int,
#      "matchStart": bool,
#      "timeRemain": float,
#      "team": int,
#      "inMatch": bool,
#    }
#
# 2. /fares?all=[True|False]
#    Returns a list of available fares:
#    [
#      {
#        "id": int,
#        "modifiers": int,
#        "src": {"x": float, "y": float},
#        "dest": {"x": float, "y": float},
#        "claimed": bool,
#        "expiry": float,
#        "pay": float,
#        "reputation": int
#      }
#    ]
#
# 3. /fares/claim/<idx>?auth=<auth>
#    Claims the fare with the given id.
#    Response payload:
#    [
#      {
#        "success": bool,
#        "message": str
#      }
#    ]
#
# 4. /fares/current/<team>
#    Returns the current fare status for your team:
#    [
#      {
#        "fare": { ... } | None,
#        "message": str
#      }
#    ]
#
# 5. /WhereAmI/<team>
#    Returns the vehicle's GPS position:
#    [
#      {
#        "position": {"x": float, "y": float},
#        "last_update": int,
#        "message": str
#      }
#    ]

def get_match_status():
    """Retrieve current match status from the VPFS server."""
    res = request.urlopen(server + f"/match?auth={authKey}")
    if res.status == 200:
        return json.loads(res.read())
    else:
        print("Error fetching match status")
        return None

def get_fares(all_fares=False):
    """
    Retrieve a list of available fares.
    Set all_fares=True to include past fares as well.
    """
    url = server + f"/fares?all={str(all_fares)}"
    res = request.urlopen(url)
    if res.status == 200:
        return json.loads(res.read())
    else:
        print("Error fetching fares")
        return []

def claim_fare(fare_id):
    """Attempt to claim a fare using its ID."""
    res = request.urlopen(server + f"/fares/claim/{fare_id}?auth={authKey}")
    if res.status == 200:
        response = json.loads(res.read())
        return response[0]['success']
    return False

def get_current_fare():
    """
    Check the current fare status using /fares/current/<team>.
    Returns the payload containing fare details and a message.
    """
    res = request.urlopen(server + f"/fares/current/{team}")
    if res.status == 200:
        return json.loads(res.read())
    else:
        print("Error fetching current fare status")
        return None

def get_position():
    """
    Get the current vehicle position from the VPFS GPS API.
    Returns a Node representing the vehicle's position.
    """
    res = request.urlopen(server + f"/WhereAmI/{team}")
    if res.status == 200:
        position_data = json.loads(res.read())
        position = position_data[0]['position']
        return Node(position['x'], position['y'])
    return None

# The GPS system uses a coordinate system with the origin in the bottom-left corner.
# Positive Y is up and positive X is to the right.
#
# The following intersections are provided in centimeters (accurate to within ±3cm).
# These can be used to refine your map grid or serve as reference points.
#
# intersections = {
#   "Aquatic Ave. & Beak St.": (452, 29),
#   "Feather St. & ?": (305, 29),       # Adjust vertical road if needed.
#   "Waddle Way & ?": (129, 29),
#   "Waterfoul Way & ?": (213, 29),
#   "Breadcrumb Ave. & The Circle": (284, 393),
#   "Waddle Way & ?": (181, 459),
#   "The Circle & Feather St.": (305, 296),
#   "Waterfoul Way & ?": (273, 307),
#   "Dabbler Dr. & Beak St.": (452, 293),
#   "The Circle": (350, 324),
#   "Mallard St. & ?": (585, 293),
#   "Drake Dr. & Beak St.": (452, 402),
#   "Mallard St. & ?": (576, 354),
#   "Duckling Dr. & Beak St.": (452, 474),
#   "Mallard St. & ?": (593, 354),
#   "Migration Ave. & Beak St.": (452, 135),
#   "Feather St. & ?": (305, 135),
#   "Mallard St. & ?": (585, 135),
#   "Quack St.": (29, 135),
#   "Waddle Way": (129, 135),
#   "Waterfoul Way": (213, 135),
#   "Pondside Ave. & Beak St.": (452, 233),
#   "Feather St. & ?": (305, 233),
#   "Mallard St. & ?": (585, 233),
#   "Quack St.": (28, 329),
#   "Waterfoul Way": (214, 241),
#   "Waddle Way": (157, 266),
#   "Tail Ave. & Beak St.": (452, 465),
#   "The Circle & Tail Ave.": (335, 387)
# }
#

if __name__ == "__main__":
    match_status = get_match_status()
    if match_status:
        print("Match Status:")
        print(f"Mode: {match_status['mode']}")
        print(f"Match Number: {match_status['match']}")
        print(f"Match Start: {match_status['matchStart']}")
        print(f"Time Remaining: {match_status['timeRemain']}")
        print(f"Team: {match_status['team']}")
        print(f"In Match: {match_status['inMatch']}")
        
        if not match_status['matchStart']:
            print("Match has not started. Exiting.")
            exit(0)
    else:
        print("Could not retrieve match status. Exiting.")
        exit(1)

    fares = get_fares(all_fares=False)
    
    for fare in fares:
        if not fare['claimed']:
            if claim_fare(fare['id']):
                print(f"Claimed fare {fare['id']}")
                
                current_fare = get_current_fare()
                if current_fare and current_fare[0]['fare'] is not None:
                    print("Current Fare Info:")
                    print(json.dumps(current_fare[0]['fare'], indent=2))
                else:
                    print("No active fare found in current fare status.")
                
                start = get_position()
                if start is None:
                    print("Could not retrieve vehicle position.")
                    break
                
                goal = Node(fare['dest']['x'], fare['dest']['y'])
                
                # Replace this with your real map grid in a full implementation.
                grid = [[0 for _ in range(100)] for _ in range(100)]
                
                path = a_star(start, goal, grid)
                if path:
                    print("Path to destination:", path)
                else:
                    print("No path found to the destination.")
                break  # Process one fare at a time.
            else:
                print(f"Failed to claim fare {fare['id']}")
        else:
            print(f"Fare {fare['id']} already claimed")
