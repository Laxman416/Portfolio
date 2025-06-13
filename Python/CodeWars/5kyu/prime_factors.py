def prime_factors(num):
    factors = []
    factor = 2
    while (num >= 2):
        if (num % factor == 0):
            factors.append(factor)
            num = num / factor
        else:
            factor += 1
    
    # number: how many times it occurs
    map = {}
    for num in factors:
        map[num] = map.get(num, 0) + 1
        
    output_string = ''
    for i in map:
        if map[i] != 1:
            string = f'({i}**{map[i]})'
        else:
            string = f'({i})'
        output_string = output_string + string
    print(output_string)
    return output_string
