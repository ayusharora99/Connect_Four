import numpy as np

NEG_INF = -9999999
POS_INF =  9999999
MAX_DEPTH = 4
MAX_PLAYER = 1
MIN_PLAYER = 2
DEPTH = 7
EXPECTIMAX_DEPTH = 4

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)


    def expectimax(self,board,depth,player_number):
        if depth == 0 or self.game_completed(board,1) or self.game_completed(board,2):
            if self.game_completed(board,1):
                
                return None, POS_INF
            elif self.game_completed(board,2):
                
                return None, NEG_INF
            else:
                
                return None, self.evaluation_function(board)

        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        valid_rows = []
        for col in valid_cols:
            row = 5
            while row > -1:
                if board[row][col] == 0:
                    valid_rows.append(row)
                    break
                row -= 1
        
        succesors = [(valid_rows[i], valid_cols[i]) for i in range(0,len(valid_rows))]

        if player_number == MAX_PLAYER:
            value = NEG_INF
            next_best_move = np.random.choice(valid_cols)
            board_copy = board.copy()
            for succ in succesors:
                board_copy[succ[0],succ[1]] = MAX_PLAYER
                new_score = self.expectimax(board_copy, depth - 1, MIN_PLAYER)[1]
                if new_score > value:
                    value = new_score
                    next_best_move = succ[1]
                board_copy[succ[0],succ[1]] = 0
            return next_best_move, value
        
        if player_number == MIN_PLAYER:
            value = 0
            next_best_move = np.random.choice(valid_cols)
            board_copy = board.copy()
            p = 1 / len(succesors)
            for succ in succesors:
                board_copy[succ[0],succ[1]] = MIN_PLAYER
                value += p * self.expectimax(board_copy, depth - 1, MAX_PLAYER)[1]
                board_copy[succ[0],succ[1]] = 0
            return next_best_move, value



    def minimax(self,board,depth, alpha, beta, player_number):
        if depth == 0 or self.game_completed(board,1) or self.game_completed(board,2):
            if self.game_completed(board,1):
                
                return None, POS_INF
            elif self.game_completed(board,2):
                
                return None, NEG_INF
            else:
                
                return None, self.evaluation_function(board)

        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        valid_rows = []
        for col in valid_cols:
            row = 5
            while row > -1:
                if board[row][col] == 0:
                    valid_rows.append(row)
                    break
                row -= 1

        succesors = [(valid_rows[i], valid_cols[i]) for i in range(0,len(valid_rows))]

        if player_number == MAX_PLAYER:
            value = NEG_INF
            next_best_move = np.random.choice(valid_cols)
            board_copy = board.copy()
            for succ in succesors:
                board_copy[succ[0],succ[1]] = MAX_PLAYER
                new_score = self.minimax(board_copy, depth - 1, alpha, beta, MIN_PLAYER)[1]
                if new_score > value:
                    value = new_score
                    next_best_move = succ[1]
                board_copy[succ[0],succ[1]] = 0
                alpha = max(alpha,value)
                if alpha >= beta:
                    break
            return next_best_move, value
        
        if player_number == MIN_PLAYER:
            value = POS_INF
            next_best_move = np.random.choice(valid_cols)
            board_copy = board.copy()
            for succ in succesors:
                board_copy[succ[0],succ[1]] = MIN_PLAYER
                new_score = self.minimax(board_copy, depth - 1, alpha, beta, MAX_PLAYER)[1]
                if new_score < value:
                    value = new_score
                    next_best_move = succ[1]
                board_copy[succ[0],succ[1]] = 0
                beta = min(beta,value)
                if alpha >= beta:
                    break
            return next_best_move, value

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        best_next_move, utility = self.minimax(board,DEPTH, NEG_INF, POS_INF, self.player_number)

        print("Next move: {} with utlity of {} ".format(best_next_move,utility))
        return best_next_move
    
    
    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        best_next_move, utility = self.expectimax(board, EXPECTIMAX_DEPTH, self.player_number)

        print("Next move: {} with utlity of {} ".format(best_next_move,utility))
        return best_next_move


    def evaluation_function(self, board):
        """
        Given the current state of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        utiliy = 0
        for row in range(0,6):
            for col in range(0,7):
                if board[row][col] == MAX_PLAYER:
                    if self.game_completed(board,1):
                        return POS_INF
                    elif self.three_in_a_row(board,1):
                        return POS_INF - 10
                    elif self.two_in_a_row(board,1):
                        return POS_INF - 20
                elif board[row][col] == MIN_PLAYER:
                    if self.game_completed(board,2):
                        return NEG_INF
                    elif self.three_in_a_row(board,2):
                        return NEG_INF + 10
                    elif self.two_in_a_row(board,2):
                        return NEG_INF + 20

    def game_completed(self, board, player_num):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def three_in_a_row(self, board, player_num):
        player_win_str = '{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def two_in_a_row(self, board, player_num):
        player_win_str = '{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))




class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

