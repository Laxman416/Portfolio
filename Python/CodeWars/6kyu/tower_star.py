def tower_builder(n_floors):
    # list of strings, spaces depending on how many floors
    # 1,3,5,7,9,11...
    tower = []
    num_spaces_each_side = get_num_spaces(n_floors)
    i = 0
    
    num_stars = 1
    
    while num_spaces_each_side >= 0:
        string_space = num_spaces_each_side * ' '
        string_star = num_stars * '*'
        full_string = string_space + string_star + string_space
        tower.append(full_string)
        num_stars = num_stars + 2
        num_spaces_each_side = num_spaces_each_side - 1
        
    print(tower)
    return tower

def get_num_spaces(num_floors):
    
    num_star = num_floors*2 -1
    num_spaces_each_side = (num_star - 1)/2
    
    return int(num_spaces_each_side)
    