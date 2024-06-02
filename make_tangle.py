


import random
import numpy as np
import numpy.random as npr

# Constants
GRID_X = 32
GRID_Y = 32
GRID_Z = 32
NUM_PATHS = 5
MIN_LENGTH = 20
MAX_LENGTH = 40
R = 15

# Directions for 3D grid movement (6 possible movements)
DIRECTIONS = [
    (1, 0, 0), (-1, 0, 0),  # +x, -x
    (0, 1, 0), (0, -1, 0),  # +y, -y
    (0, 0, 1), (0, 0, -1)   # +z, -z
]

#DIRECTIONS = [np.array(d, dtype=np.float32) for d in DIRECTIONS]
#print(DIRECTIONS)

def generate_random_path(grid, idx):
    # Random start point
    while True:
        start_x = random.randint(0, GRID_X - 1)
        start_y = random.randint(0, GRID_Y - 1)
        start_z = random.randint(0, GRID_Z - 1)
        if (start_x-GRID_X//2)**2 + (start_y-GRID_Y//2)**2 + (start_z-GRID_Z//2)**2 < (R-3)**2:
            break
        
    # Random length of the path
    path_length = random.randint(MIN_LENGTH, MAX_LENGTH)
    
    path = [(start_x, start_y, start_z)]
    #grid[start_x, start_y, start_z] = idx
    
    direction= None
    momentum = 3.0
    for _ in range(path_length - 1):
        # Get the last point
        last_x, last_y, last_z = path[-1]
        

        if direction is not None:

            direction_weights = []
            for d in DIRECTIONS:
                r = (last_x+d[0]-GRID_X//2)**2 + (last_y+d[1]-GRID_Y//2)**2 + (last_z+d[2]-GRID_Z//2)**2
                e = -np.exp(0.01*(r-R*R*0.8))
                if not(0<=last_x+d[0]<GRID_X and 0<=last_y+d[1]<GRID_Y and 0<=last_z+d[2]<GRID_Z) or grid[last_x+d[0], last_y+d[1], last_z+d[2]]:
                    direction_weights.append(-10000)
                else:
                    direction_weights.append(momentum*(np.clip(np.dot(direction_ema, np.array(d)), -1, 0.2)) + e)

            direction_weights = momentum*np.array(direction_weights)
            # Also push away from edges
                        

            
            direction_weights = direction_weights +npr.randn(len(direction_weights))
            #direction_weights /= direction_weights.sum()
            
            idx = np.argmax(direction_weights)
            direction = DIRECTIONS[idx] #random.choices(DIRECTIONS, weights = direction_weights)[0]

            direction_ema = 0.8*direction_ema + 0.2*np.array(direction)
                
        else:
            direction = random.choice(DIRECTIONS)
            direction_ema = np.array(direction, dtype=np.float32)
        
        # Calculate new point
        new_x = last_x + direction[0]
        new_y = last_y + direction[1]
        new_z = last_z + direction[2]
        
        # Ensure the new point is within the grid bounds
        if 0 <= new_x < GRID_X and 0 <= new_y < GRID_Y and 0 <= new_z < GRID_Z:
            path.append((new_x, new_y, new_z))
            grid[new_x, new_y, new_z] = idx
        else:
            # If out of bounds, try another direction or stop growing this path
            continue
    
    return path

def make_tangle(i):
    npr.seed(i)
    random.seed(i)
# Generate paths
    grid = np.zeros((GRID_X,GRID_Y,GRID_Z), dtype=np.int8)
    paths = [generate_random_path(grid, i+1) for i in range(NUM_PATHS)]

    grid_2d = np.zeros((GRID_X, GRID_Y), dtype=np.int8)
    depth_2d = np.zeros((GRID_X, GRID_Y), dtype=np.int8)
    depth_2d[:,:] = GRID_Z
    output = np.zeros((GRID_X, GRID_Y, 5), dtype=np.int8)
    for i, p in enumerate(paths):
        for x, y, z in p:
            if z<depth_2d[x,y]:
                depth_2d[x,y] = z
                grid_2d[x,y] = i+1
            output[x,y,i] = 1

    #return output.transpose((2,0,1)).astype(np.float32), np.stack([grid_2d, depth_2d/GRID_Z], axis=0).astype(np.float32)
    return output.astype(np.float32), np.stack([grid_2d, depth_2d/GRID_Z], axis=2).astype(np.float32)

if __name__=="__main__":
    x, y = make_tangle(npr.randint(100))
    import matplotlib.pyplot as plt
    plt.imshow(y[0])
    plt.figure()
    plt.imshow((y[0]>0)*y[1])
    plt.show()

        


