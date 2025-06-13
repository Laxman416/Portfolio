# In this kata, your task is to create all permutations of a non-empty input string and remove duplicates, if present.

import itertools


def permutations(s):
    # n choose n permuation

    permutations = set(itertools.permutations(s))
    result = []
    
    for p in permutations:
        word = ''.join(p)
        print(word)
        result.append(word)
        
    return result
              


    