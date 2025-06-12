"""
Each circle is represented as an array of three elements. The first two elements are the coordinates of the center, the third is the radius.

Example
For c1 = [2, 2, 3] and c2 = [0, -1, 3], the output should be 8 (see the picture below)
"""
import numpy as np

def overlap_point(c1, c2):
    # If point lies within a circle it will satisfy the equation (x-a)**2 + (y-a)**2 < r**2"
    
    x_min, x_max, y_min, y_max = area_to_check(c1,c2)
    coordinates_to_check = []
    
    x = x_min

    while x >= x_min and x <= x_max:
        y = y_min 
        while y >= y_min and y <= y_max:
            coordinates_to_check.append([x,y])
            y = y + 1
        x = x + 1
            
    allowed_points = []
    
    for coordinate in coordinates_to_check:
        if check_point_in_circle(c1, coordinate) and check_point_in_circle(c2, coordinate):
            allowed_points.append(coordinate)
        
    return len(allowed_points)

def check_point_in_circle(circle, point):
    radius = circle[2]
    radius_squared = radius**2
    
    if (point[0]-circle[0])**2 + (point[1]-circle[1])**2 <= radius_squared:
        return True
    else:
        return False

def area_to_check(c1,c2):
    if c1[0]-c1[2] < c2[0]-c2[2]:
        x_min = int(np.floor(c1[0]-c1[2]))
    else:
        x_min = int(np.floor(c2[0]-c2[2]))
        
    if c1[0]+c1[2] > c2[0]+c2[2]:
        x_max = int(np.ceil(c1[0]+c1[2]))
    else:
        x_max = int(np.ceil(c2[0]+c2[2]))
        
    if c1[1]-c1[2] < c2[1]-c2[2]:
        y_min = int(np.floor(c1[1]-c1[2]))
    else:
        y_min = int(np.floor(c2[1]-c2[2]))
        
    if c1[1]+c1[2] > c2[1]+c2[2]:
        y_max = int(np.ceil(c1[1]+c1[2]))
    else:
        y_max = int(np.ceil(c2[1]+c2[2]))
        
    return x_min, x_max, y_min, y_max
    
    