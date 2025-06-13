"""
Given two different positions on a chess board, find the least number of moves it would take a knight to get from one to the other. The positions will be passed as two arguments in algebraic notation. For example, knight("a3", "b5") should return 1.

The knight is not allowed to move off the board. The board is 8x8.
"""

def knight(p1, p2):
    
    allowed_moves = [p1]
    allowed_letters = ['a','b','c','d','e','f','g','h']
    allowed_numbers = ['1','2','3','4','5','6','7','8']
    least_number_of_moves = 0
    find_all_knight_moves(allowed_moves, allowed_letters, allowed_numbers)
    
    while p2 not in allowed_moves:
        allowed_moves = find_all_knight_moves(allowed_moves, allowed_letters, allowed_numbers)
        least_number_of_moves = least_number_of_moves + 1    
        
    print(f'Least number of moves: {least_number_of_moves}')
    return least_number_of_moves

def find_all_knight_moves(array_of_positions, allowed_letters, allowed_numbers):
    # Finds all knight move postions given a intial postion
    allowed_knight_moves = []
    
    allowed_moves_dict = {
        1: [1,2],
        2: [1,-2],
        3: [-1,2],
        4: [-1,-2],
        5: [-2,-1],
        6: [-2,1],
        7: [2,-1],
        8: [2,1]}
    
    for i in array_of_positions:
        for j in allowed_moves_dict:
            letter_position = chr(ord(i[0]) + allowed_moves_dict[j][0])
            number_position = str(int(i[1]) + allowed_moves_dict[j][1])
            
            if letter_position in allowed_letters:
                if number_position in allowed_numbers:
                    position = letter_position + number_position
                    allowed_knight_moves.append(position)
                        
    return allowed_knight_moves


        