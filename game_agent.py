"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    reflect_score = reflect(game, player)
    attack_opponent_score = attack_opponent(game, player)
    survive_score = survive(game, player)

    return survive_score

def reflect(game, player):
    """
    Heuristic prioritising the reflection of the opponent

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Get opponent position
    opponent_position = game.get_player_location(game.get_opponent(player))
    #Get the legal moves for the current player
    legal_moves = game.get_legal_moves(player)
    #Get the reflecting move of the the opponent
    reflect_opponent = (game.height-opponent_position[0]-1, game.width-opponent_position[1]-1)
    #If the reflecting move is legal return infinity to prioritise this move
    if reflect_opponent in legal_moves:
        return float("inf")
    return float(game.utility(player))

def attack_opponent(game, player):
    """
    Heuristic prioritising the moves that contain future moves common to the
    opponents legal moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Get opponent moves
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    #Get legal moves for the current player
    legal_moves = game.get_legal_moves(player)
    #Get the current utility, then use common moves as a bonus
    count = game.utility(player)
    for move in legal_moves:
        if move in opponent_moves:
            count += 1
    return float(count)

def survive(game, player):
    """
    Heuristic prioritising survival by staying on the side/quadrant
    with the most open blocks.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #get the current board
    board_state = game.__board_state__
    #get the mid point to determine the different quadrants
    middlex = math.floor((game.width-1)/2)
    middley = math.floor((game.height-1)/2)
    left_section = 0
    right_section = 0
    top_section = 0
    bottom_section = 0
    #for each row and col in row determine the number of used block in each quadrant
    for idx, row in enumerate(board_state):
        for idxC, col in enumerate(row):
            if col > 0:
                if idxC < middlex:
                    left_section+=1
                else:
                    right_section+=1
                if idx < middley:
                    top_section+=1
                else:
                    bottom_section+=1
    """Get the current player location and determine which is the best
       quadrant by returning the section with minimum number of used blocks.
    """
    player_location = game.get_player_location(player)
    if player_location[0] < middlex:
        if player_location[1] < middley:
            return float(min(left_section, top_section))
        else:
            return float(min(left_section, bottom_section))
    else:
        if player_location[1] < middley:
            return float(min(right_section, top_section))
        else:
            return float(min(right_section, bottom_section))
                


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        move = (-1, -1)
        if len(legal_moves) == 0:
            return move
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            #if iterative follow the logic of increasing the depth at each iteration
            if self.iterative:
                depth = 0
                while True:
                    if self.method == 'minimax':
                       score, move = self.minimax(game, depth)
                    else:
                       score,move = self.alphabeta(game, depth)

                    depth+=1
            else:
                if self.method == 'minimax':
                    score,move = self.minimax(game, self.search_depth)
                else:
                    score,move = self.alphabeta(game, self.search_depth)
        except Timeout:
            # Handle any actions required at timeout, if necessary
            #Return the most recent best move
            return move

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

           function MINIMAX-DECISION(state) returns an action
             return arg max a ∈ ACTIONS(s) MIN-VALUE(RESULT(state, a))

            aimacode (2017).
            Minimax-Decision
            [online] Github.
            Available at:
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
           [Accessed 06 Mar. 2017].

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #Throw timeout            
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        #Test for terminal leaf node
        if self.terminal_test(game, depth):
            return self.score(game, self), game.get_player_location(self)

        #init with invalid move and worst score
        final_score = float("-inf")
        final_move = (-1,-1)
        try:
            #get the legal moves
            legal_moves = game.get_legal_moves()
            """for each move get the result of minimax min for
               a decreasing depth. If the result is better than the previous
               best, assign the best as the result.
            """
            for move in legal_moves:
                result = self.minimax_min(game.forecast_move(move), depth-1)
                if result[0] > final_score:
                    final_score = result[0]
                    final_move = move
        except Timeout:
            return final_score, final_move

        return final_score, final_move
        


    def minimax_min(self, game, depth):
        """Implement the minimax search algorithm as described in the lectures.

            function MIN-VALUE(state) returns a utility value
             if TERMINAL-TEST(state) the return UTILITY(state)
             v ← ∞
             for each a in ACTIONS(state) do
               v ← MIN(v, MAX-VALUE(RESULT(state, a)))
             return v

          aimacode (2017).
            Minimax-Decision
            [online] Github.
            Available at:
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
           [Accessed 06 Mar. 2017].

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #Raise timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        #check for terminal leaf node
        if self.terminal_test(game, depth):
            return self.score(game, self), game.get_player_location(self)
        #set invalid move and best possible score
        score = float("inf")
        move = (-1,-1)
        legal_moves = game.get_legal_moves()
        for m in legal_moves:
            """for each move get the result of minimax max for
               a decreasing depth. If the result is better than the previous
               best, assign the best as the result.
            """
            result = self.minimax_max(game.forecast_move(m), depth-1)
            if result[0] < score:
                score = result[0]
                move = m
            
        return score, move

    def minimax_max(self, game, depth):
        """Implement the minimax search algorithm as described in the lectures.

              function MAX-VALUE(state) returns a utility value
             if TERMINAL-TEST(state) the return UTILITY(state)
             v ← −∞
             for each a in ACTIONS(state) do
               v ← MAX(v, MIN-VALUE(RESULT(state, a)))
             return v

          aimacode (2017).
            Minimax-Decision
            [online] Github.
            Available at:
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
           [Accessed 06 Mar. 2017].

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #raise timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        #check for terminal leaf node
        if self.terminal_test(game, depth):
            return self.score(game, self), game.get_player_location(self)
        #set the worst move and score
        score = float("-inf")
        move = (-1, -1)
        legal_moves = game.get_legal_moves()
        for m in legal_moves:
            """for each move get the result of minimax max for
               a decreasing depth. If the result is better than the previous
               best, assign the best as the result.
            """
            result = self.minimax_min(game.forecast_move(m), depth-1)
            if result[0] > score:
                score = result[0]
                move = m
        return score, move

    def terminal_test(self, game, depth):
        """
            If the number of legal moves is 0 or the depth is zero
            treat the node as a terminal leaf node
        """
        if len(game.get_legal_moves()) == 0 or depth == 0:
            return True
        return False

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        function ALPHA-BETA-SEARCH(state) returns an action
         v ← MAX-VALUE(state, −∞, +∞)
         return the action in ACTIONS(state) with value v

            aimacode (2017).
             ALPHA-BETA-SEARCH
            [online] Github.
            Available at:
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
           [Accessed 06 Mar. 2017].

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #raise timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        # Set the worst move and score
        final_score = float("-inf")
        final_move = (-1,-1)
        #return the result of maximising alpha beta
        return self.alphabeta_max(game, depth, alpha, beta)
        

    def alphabeta_min(self, game, depth, alpha, beta):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

         function MIN-VALUE(state, α, β) returns a utility value
         if TERMINAL-TEST(state) the return UTILITY(state)
         v ← +∞
         for each a in ACTIONS(state) do
           v ← MIN(v, MAX-VALUE(RESULT(state, a), α, β))
           if v ≤ α then return v
           β ← MIN(β, v)
         return v

            aimacode (2017).
            ALPHA-BETA-SEARCH
            [online] Github.
            Available at:
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
           [Accessed 06 Mar. 2017].

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #raise timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            return self.score(game, self), game.get_player_location(self)
        #check for terminal leaf node
        if self.terminal_test(game, depth):
            return self.score(game, self), game.get_player_location(self)
        #set the worst move and best score
        score = float("inf")
        move = (-1,-1)
        legal_moves = game.get_legal_moves()
        """for each move get the result of alphabeta_max for
               a decreasing depth. If the result is better than the previous
               best, assign the best as the result.

               Update alpha and beta as described by the above algorithm
        """
        for m in legal_moves:
            result = self.alphabeta_max(game.forecast_move(m), depth-1, alpha, beta)
            if result[0] < score:
                score = result[0]
                move = m
            if score <= alpha:
                return score, move
            beta = min(beta, score)
            
        return score, move

    def alphabeta_max(self, game, depth, alpha, beta):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

         function MIN-VALUE(state, α, β) returns a utility value
         if TERMINAL-TEST(state) the return UTILITY(state)
         v ← +∞
         for each a in ACTIONS(state) do
           v ← MIN(v, MAX-VALUE(RESULT(state, a), α, β))
           if v ≤ α then return v
           β ← MIN(β, v)
         return v

            aimacode (2017).
            ALPHA-BETA-SEARCH
            [online] Github.
            Available at:
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
           [Accessed 06 Mar. 2017].

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #raise timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            return self.score(game, self), game.get_player_location(self)
        #check for terminal leaf node
        if self.terminal_test(game, depth):
            return self.score(game, self), game.get_player_location(self)
        #set worst move and score
        score = float("-inf")
        move = (-1, -1)
        legal_moves = game.get_legal_moves()
        """for each move get the result of alphabeta_max for
               a decreasing depth. If the result is better than the previous
               best, assign the best as the result.

               Update alpha and beta as described by the above algorithm
        """
        for m in legal_moves:
            result = self.alphabeta_min(game.forecast_move(m), depth-1, alpha, beta)
            if result[0] > score:
                score = result[0]
                move = m
            if score >= beta:
                return score, move
            alpha = max(alpha, score)
        return score, move
