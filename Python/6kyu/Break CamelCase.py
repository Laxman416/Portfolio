"""
Break CamelCase 6kyu

Complete the solution so that the function will break up camel casing, using a space between words.

"""
def solution(s):
    newStr = ""
    for letter in s:
        if letter.isupper():
            newStr += " "
        newStr += letter
    return newStr