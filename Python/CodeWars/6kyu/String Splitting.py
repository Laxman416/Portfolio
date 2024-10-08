"""
String Manipulation 6kyu

Complete the solution so that it splits the string into pairs of two characters. If the string contains an odd number of characters then it should replace the missing second character of the final pair with an underscore ('_').

Examples:

* 'abc' =>  ['ab', 'c_']
* 'abcdef' => ['ab', 'cd', 'ef']

"""

def solution(string):
    list = []
    length = len(string)
    i = 0
    while i < length:
        if i + 1 < length:  # Check if there's a next character
            list.append(string[i:i+2])  
        else:
            list.append(f'{string[i]}_')
        i += 2
    return list