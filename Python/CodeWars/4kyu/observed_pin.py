import itertools

def get_pins(observed):
    """
    horrizontal or diagnal
    
    """
    map = {
    '1': ['1', '2', '4'],
    '2': ['1', '2', '3', '5'],
    '3': ['2', '3', '6'],
    '4': ['1', '4', '5', '7'],
    '5': ['2', '4', '5', '6', '8'],
    '6': ['3', '5', '6', '9'],
    '7': ['4', '7', '8'],
    '8': ['5', '7', '8', '9', '0'],
    '9': ['6', '8', '9'],
    '0': ['8', '0']
        }
    
    observed_combination_2d_array = []
    for i in observed:
        observed_combination_2d_array.append(map[i])
    
    # set to remove duplicates
    all_combinations_joined = set()

    for combo in itertools.product(*observed_combination_2d_array):
        joined = ''.join(combo)
        if joined not in all_combinations_joined:
            all_combinations_joined.add(joined)

    return all_combinations_joined

