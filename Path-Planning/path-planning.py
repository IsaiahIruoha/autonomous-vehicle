#!/usr/bin/env python3
import json
import math
import time
from urllib import request

# VPFS server configuration; update these as needed for lab/home/competition.
SERVER_IP = "127.0.0.1"  # placeholder IP; change to the actual server IP
SERVER = f"http://{SERVER_IP}:5000"
AUTH_KEY = "40"          # For lab testing, your team number is used as auth
TEAM = 40

# ---------------- Graph Definition ----------------
# Sample graph representation of intersections.
# Coordinates are in meters. (You might want to convert from centimeters.)
# Each node is given a coordinate and a list of neighboring nodes with associated cost.
graph = {
    # --- Vertical Road: Beak St. ---
    "A": {  # Aquatic Ave. & Beak St. (452, 29 → (4.52, 0.29))
        "coord": (4.52, 0.29),
        "neighbors": {
            "P": 1.06  # A <-> P
        }
    },
    "P": {  # Migration Ave. & Beak St. (452, 135 → (4.52, 1.35))
        "coord": (4.52, 1.35),
        "neighbors": {
            "A": 1.06,
            "V": 0.98
        }
    },
    "V": {  # Pondside Ave. & Beak St. (452, 233 → (4.52, 2.33))
        "coord": (4.52, 2.33),
        "neighbors": {
            "P": 0.98,
            "I": 0.60
        }
    },
    "I": {  # Dabbler Dr. & Beak St. (452, 293 → (4.52, 2.93))
        "coord": (4.52, 2.93),
        "neighbors": {
            "V": 0.60,
            "L": 1.09
        }
    },
    "L": {  # Drake Dr. & Beak St. (452, 402 → (4.52, 4.02))
        "coord": (4.52, 4.02),
        "neighbors": {
            "I": 1.09,
            "AB": 0.63
        }
    },
    "AB": {  # Tail Ave. & Beak St. (452, 465 → (4.52, 4.65))
        "coord": (4.52, 4.65),
        "neighbors": {
            "L": 0.63,
            "N": 0.09
        }
    },
    "N": {  # Duckling Dr. & Beak St. (452, 474 → (4.52, 4.74))
        "coord": (4.52, 4.74),
        "neighbors": {
            "AB": 0.09
        }
    },

    # --- Vertical Road: Feather St. ---
    "B": {  # Feather St. & (assumed) vertical Feather (305, 29 → (3.05, 0.29))
        "coord": (3.05, 0.29),
        "neighbors": {
            "Q": 1.06
        }
    },
    "Q": {  # (Feather St., 305, 135 → (3.05, 1.35))
        "coord": (3.05, 1.35),
        "neighbors": {
            "B": 1.06,
            "W": 0.98
        }
    },
    "W": {  # (Feather St., 305, 233 → (3.05, 2.33))
        "coord": (3.05, 2.33),
        "neighbors": {
            "Q": 0.98,
            "G": 0.63
        }
    },
    "G": {  # The Circle & Feather St. (305, 296 → (3.05, 2.96))
        "coord": (3.05, 2.96),
        "neighbors": {
            "W": 0.63
        }
    },

    # --- Vertical Road: Quack St. ---
    "S": {  # Quack St. (29, 135 → (0.29, 1.35))
        "coord": (0.29, 1.35),
        "neighbors": {
            "Y": 1.94
        }
    },
    "Y": {  # Quack St. (28, 329 → (0.28, 3.29))
        "coord": (0.28, 3.29),
        "neighbors": {
            "S": 1.94
        }
    },

    # --- Horizontal Road: Waddle Way ---
    "C": {  # Waddle Way (129, 29 → (1.29, 0.29))
        "coord": (1.29, 0.29),
        "neighbors": {
            "T": 1.06
        }
    },
    "T": {  # Waddle Way (129, 135 → (1.29, 1.35))
        "coord": (1.29, 1.35),
        "neighbors": {
            "C": 1.06,
            "AA": 1.34
        }
    },
    "AA": {  # Waddle Way (157, 266 → (1.57, 2.66))
        "coord": (1.57, 2.66),
        "neighbors": {
            "T": 1.34,
            "F": 1.95
        }
    },
    "F": {  # Waddle Way (181, 459 → (1.81, 4.59))
        "coord": (1.81, 4.59),
        "neighbors": {
            "AA": 1.95
        }
    },

    # --- Horizontal Road: Waterfoul Way. ---
    "D": {  # Waterfoul Way. (213, 29 → (2.13, 0.29))
        "coord": (2.13, 0.29),
        "neighbors": {
            "U": 1.06
        }
    },
    "U": {  # Waterfoul Way. (213, 135 → (2.13, 1.35))
        "coord": (2.13, 1.35),
        "neighbors": {
            "D": 1.06,
            "Z": 1.06
        }
    },
    "Z": {  # Waterfoul Way. (214, 241 → (2.14, 2.41))
        "coord": (2.14, 2.41),
        "neighbors": {
            "U": 1.06,
            "H": 0.89
        }
    },
    "H": {  # Waterfoul Way. (273, 307 → (2.73, 3.07))
        "coord": (2.73, 3.07),
        "neighbors": {
            "Z": 0.89
        }
    },

    # --- Horizontal Road: Mallard St. ---
    "R": {  # Mallard St. (585, 135 → (5.85, 1.35))
        "coord": (5.85, 1.35),
        "neighbors": {
            "X": 0.98
        }
    },
    "X": {  # Mallard St. (585, 233 → (5.85, 2.33))
        "coord": (5.85, 2.33),
        "neighbors": {
            "R": 0.98,
            "K": 0.60
        }
    },
    "K": {  # Mallard St. (585, 293 → (5.85, 2.93))
        "coord": (5.85, 2.93),
        "neighbors": {
            "X": 0.60,
            "M": 0.62,
            "O": 0.62
        }
    },
    "M": {  # Mallard St. (576, 354 → (5.76, 3.54))
        "coord": (5.76, 3.54),
        "neighbors": {
            "K": 0.62,
            "O": 0.17
        }
    },
    "O": {  # Mallard St. (593, 354 → (5.93, 3.54))
        "coord": (5.93, 3.54),
        "neighbors": {
            "K": 0.62,
            "M": 0.17
        }
    },

    # --- Horizontal Road: The Circle ---
    "J": {  # The Circle (350, 324 → (3.50, 3.24))
        "coord": (3.50, 3.24),
        "neighbors": {
            "AC": 0.65
        }
    },
    "AC": {  # The Circle (335, 387 → (3.35, 3.87))
        "coord": (3.35, 3.87),
        "neighbors": {
            "J": 0.65,
            "E": 0.51
        }
    },
    "E": {  # Breadcrumb Ave. & The Circle (284, 393 → (2.84, 3.93))
        "coord": (2.84, 3.93),
        "neighbors": {
            "AC": 0.51
        }
    },

    # --- Single intersections (no same-road neighbor) ---
    "K": {  # Dabbler Dr. is represented by I (already in Beak St. group)
        # Already defined as I.
    },
    # Nodes with unique horizontal road names (only one intersection on that road)
    # Aquatic Ave.: A already defined.
    # Dabbler Dr.: I already defined.
    # Drake Dr.: L already defined.
    # Duckling Dr.: N already defined.
    # Migration Ave.: P already defined.
    # Pondside Ave.: V already defined.
    # Tail Ave.: AB already defined.
}


# ---------------- A* Path Planning ----------------
def heuristic(node, goal):
    """Euclidean distance between two nodes as the heuristic."""
    x1, y1 = graph[node]["coord"]
    x2, y2 = graph[goal]["coord"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def a_star(start, goal):
    """Computes the shortest path from start to goal using A*."""
    open_set = {start}
    came_from = {}
    g_score = {node: float("inf") for node in graph}
    g_score[start] = 0
    f_score = {node: float("inf") for node in graph}
    f_score[start] = heuristic(start, goal)

    while open_set:
        # Choose the node with the lowest f_score.
        current = min(open_set, key=lambda node: f_score[node])
        if current == goal:
            # Reconstruct the path.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        open_set.remove(current)
        for neighbor, cost in graph[current]["neighbors"].items():
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)
    return None

# ---------------- VPFS API Communication ----------------
def get_current_position(team):
    """Fetches the current vehicle position from VPFS."""
    url = SERVER + f"/WhereAmI/{team}"
    with request.urlopen(url) as res:
        if res.status == 200:
            data = json.loads(res.read())
            pos = data[0]["position"]
            return pos["x"], pos["y"]
        else:
            print("Error fetching current position.")
            return None

def get_available_fares():
    """Retrieves the list of available fares."""
    url = SERVER + "/fares"
    with request.urlopen(url) as res:
        if res.status == 200:
            return json.loads(res.read())
        else:
            print("Error fetching fares.")
            return None

def claim_fare(fare_id):
    """Attempts to claim a fare using its ID."""
    url = SERVER + f"/fares/claim/{fare_id}?auth={AUTH_KEY}"
    with request.urlopen(url) as res:
        if res.status == 200:
            data = json.loads(res.read())
            return data["success"]
        else:
            print("Error claiming fare.")
            return False

def get_current_fare(team):
    """Gets the current active fare for the team."""
    url = SERVER + f"/fares/current/{team}"
    with request.urlopen(url) as res:
        if res.status == 200:
            data = json.loads(res.read())
            return data["fare"]
        else:
            print("Error fetching current fare.")
            return None

# ---------------- Utility ----------------
def nearest_node(x, y):
    """Returns the nearest graph node to the (x,y) position."""
    best_node = None
    best_dist = float("inf")
    for node, info in graph.items():
        cx, cy = info["coord"]
        dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_node = node
    return best_node

# ---------------- Main Path Planning Logic ----------------
def main():
    print("=== Autonomous Taxi Path Planning ===")
    # Get current vehicle position
    pos = get_current_position(TEAM)
    if pos is None:
        return
    print("Current position (x, y):", pos)

    # Get list of fares and claim the first unclaimed one
    fares = get_available_fares()
    if fares is None:
        return
    chosen_fare = None
    for fare in fares:
        if not fare["claimed"]:
            chosen_fare = fare
            break
    if chosen_fare is None:
        print("No available fare to claim.")
        return

    if not claim_fare(chosen_fare["id"]):
        print("Failed to claim fare:", chosen_fare["id"])
        return
    print("Fare claimed successfully, ID:", chosen_fare["id"])

    # Get detailed fare information
    fare_details = get_current_fare(TEAM)
    if fare_details is None:
        print("No active fare found.")
        return

    pickup = fare_details["src"]
    dropoff = fare_details["dest"]
    print("Fare Pickup (x, y):", (pickup["x"], pickup["y"]))
    print("Fare Dropoff (x, y):", (dropoff["x"], dropoff["y"]))

    # Map the positions to the nearest nodes in our graph.
    current_node = nearest_node(*pos)
    pickup_node = nearest_node(pickup["x"], pickup["y"])
    dropoff_node = nearest_node(dropoff["x"], dropoff["y"])

    print("Mapped current position to node:", current_node)
    print("Mapped pickup position to node:", pickup_node)
    print("Mapped dropoff position to node:", dropoff_node)

    # Compute the route from current position to pickup
    route_to_pickup = a_star(current_node, pickup_node)
    if route_to_pickup is None:
        print("No route found to pickup location.")
        return
    print("Planned route to pickup:", route_to_pickup)

    # Compute the route from pickup to dropoff
    route_to_dropoff = a_star(pickup_node, dropoff_node)
    if route_to_dropoff is None:
        print("No route found to dropoff location.")
        return
    print("Planned route to dropoff:", route_to_dropoff)

    # Combine routes (avoid duplicate pickup node)
    overall_route = route_to_pickup + route_to_dropoff[1:]
    print("Overall planned route:", overall_route)

    # In an actual vehicle, the computed route would now be used to generate control commands.
    # For this example, we only print the route.

if __name__ == "__main__":
    main()
