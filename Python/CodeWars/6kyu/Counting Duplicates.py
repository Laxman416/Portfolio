"""" Count the number of Duplicates

6kyu Count() function

Write a function that will return the count of distinct case-insensitive alphabetic characters and numeric digits that occur more than once in the input string. The input string can be assumed to contain only alphabets (both uppercase and lowercase) and numeric digits.

Example
"abcde" -> 0 # no characters repeats more than once
"aabbcde" -> 2 # 'a' and 'b'
"aabBcde" -> 2 # 'a' occurs twice and 'b' twice (`b` and `B`)
"indivisibility" -> 1 # 'i' occurs six times
"Indivisibilities" -> 2 # 'i' occurs seven times and 's' occurs twice
"aA11" -> 2 # 'a' and '1'
"ABBA" -> 2 # 'A' and 'B' each occur twice
"""
from collections import Counter

def duplicate_count(text):
    duplicated_list = []
    character_count = {}
    text = text.lower()
    
    character_count = Counter(text)
    
    for key in character_count:
        if character_count[key] > 1:
            duplicated_list.append(key)

    duplicated_count_value = len(duplicated_list)
    
    return duplicated_count_value