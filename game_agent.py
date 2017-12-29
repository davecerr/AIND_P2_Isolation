
# coding: utf-8

# In[ ]:

# %load game_agent.py
"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def wall_checker(move, walls):
    for wall in walls:
        if move in wall:
            return True
    return False

def corner_checker(move, corners):
    for corner in corners:
        if move in corner:
            return True
    return False

def percent_occupation(game):
    return int((game.move_count/(game.width * game.height))*100)

def corners_score(game, player):
    corners = [(0,0), (0,game.width-1), (game.height-1,0), (game.height-1,game.width-1)]
    
    # Heavy reward and penalty for ultimate game states
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    # Tracking scores
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_score = 0
    opp_score = 0
    own_flexibility = 0
    opp_flexibility = 0
    
    # Award either player for going in corners when <60% occupied and penalise heavily for doing so when >60% occupied
    for move in own_moves:
        if corner_checker(move, corners) == True and (percent_occupation(game) < 60):
            own_score += 15
        elif corner_checker(move, corners) == True and (percent_occupation(game) > 60):
            own_score -= 40
        else:
            own_flexibility += 10
    for move in opp_moves:
        if corner_checker(move, corners) == True and (percent_occupation(game) < 60):
            opp_score += 15
        elif corner_checker(move, corners) == True and (percent_occupation(game) > 60):
            opp_score -= 40
        else:
            opp_flexibility += 10
    
    # Award a score that rewards us for making more good corner moves than our opponent and for playing non-corner moves
    # that leave us more flexible
    return float((own_score - opp_score) + (own_flexibility - opp_flexibility))

def walls_score(game, player):
    
    walls =  [[(0,i) for i in range(game.width)],
             [(i,0) for i in range(game.height)],
             [(game.height-1,i) for i in range(game.width)],
             [(i,game.width-1) for i in range(game.height)]]
    
    # Heavy reward and penalty for ultimate game states
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    # Tracking scores
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_score = 0
    opp_score = 0
    own_flexibility = 0
    opp_flexibility = 0
    
    # Award either player for going in corners when <60% occupied and penalise heavily for doing so when >60% occupied
    for move in own_moves:
        if wall_checker(move, walls) == True and (percent_occupation(game) < 50):
            own_score += 10
        elif wall_checker(move, walls) == True and (percent_occupation(game) > 50) and (percent_occupation(game) < 85):
            own_score -= 20
        elif wall_checker(move, walls) == True and (percent_occupation(game) > 85):  
            own_score -= 30
        else:
            own_flexibility += 5
    for move in opp_moves:
        if wall_checker(move, walls) == True and (percent_occupation(game) < 50):
            opp_score += 10
        elif wall_checker(move, walls) == True and (percent_occupation(game) > 50) and (percent_occupation(game) < 85):
            opp_score -= 20
        elif wall_checker(move, walls) == True and (percent_occupation(game) > 85):  
            opp_score -= 30
        else:
            opp_flexibility += 5
    
    # Award a score that rewards us for making more good wall moves than our opponent and for playing non-wall moves
    # that leave us more flexible
    return float((own_score - opp_score) + (own_flexibility - opp_flexibility))

def wall_and_corner_score(game, player):
    corner_score = corners_score(game, player)
    wall_score = walls_score(game, player)
    return float(0.3*wall_score + 0.7*corner_score)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    # This heuristic is an adjustment to the improved_score heuristic from the lectures whereby the agent's reward is 
    # weighted by progression of the game. Early on truns is low and the player is rewarded for playing the safety strategy
    # i.e. maintaining distance from opponent to avoid being isolated. Later, turns increases and they are rewarded
    # for aggresively trying to maintain as many moves for themselves whilst isolating the opponent. There is also a penalty 
    # for going near the centre - this starts out very low (i.e. encourage to control centre ground at start) but then 
    # rapidly increases (power of 3) to encourage playing further away as game progresses.
    return wall_and_corner_score(game, player)
    


def custom_score_2(game, player):
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
   
    # This heuristic will try to force a board partition early on by awarding a high score to those boxes close to 
    # the main vertical and horizontal
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float(abs(h - y) + abs(w - x))
    


def custom_score_3(game, player):
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
    
    # This heuristic is an adjustment to the improved_score heuristic from the lectures whereby the agent's reward is 
    # weighted by an aggression factor that is close to 0 at the start of the game and close to 1 at the end of the game
    # (essentially this is the opposite behaviour to custom_score function above). This means the agent is rewarded for
    # keeping their moves open at the start (avoiding early isolation) and then at the end they are rewarded for 
    # aggressively trying to isolate their opponent.
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    aggression = (25 - len(game.get_legal_moves())) / 25 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - aggression * opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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
        

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # For minimax we don't implement iterative deepening. search_depth is initialised to 3 and so our agent only 
            # explores minimax scoring to 3 levels deeper in the tree in order to make a decision. In our Aplhabeta agent 
            # below we implement iterative deepening - it will go as deep as possible in the allowed time.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    

        
        
    def min_value(self, game, depth):
        # Check for timeout in every call to helper functions 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # If no available moves then return the terminal utility/score
        if not game.get_legal_moves():
            return self.score(game, self)
                
        # Once minimax has been recursively applied enough times the depth countdown hits zero and we return 
        # the utility/score at that particular depth limitation to then be propagated back up the tree
        if depth == 0:
            return self.score(game, self)
                
        # Initially set v as worst possible result for MIN player then consider every possible move. As these moves
        # backpropagate up a layer, the depth counter is reduced by 1 (terminates when hits zero) and the move we
        # chose is passed to a MAX node. v is keeping track of our current BEST possible play by the MIN player
        # and this should be updated to the minimum of the "old v" and the output of a MAX player's node that our
        # previous MIN player's choice of move m fed into.
        v = float('inf') 
        for m in game.get_legal_moves():
            # Call board evaluation using self.score i.e. custom scoring
            v = min(v, self.max_value(game.forecast_move(m), depth-1))
        return v
        
    def max_value(self, game, depth):
        # Check for timeout in every call to helper functions 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
                
        # If no available moves then return the terminal utility/score
        if not game.get_legal_moves():
            return self.score(game, self)
                
        # Once minimax has been recursively applied enough times the depth countdown hits zero and we return 
        # the utility/score at that particular depth limitation to then be propagated back up the tree
        if depth == 0:
            return self.score(game, self)
                
        # Initially set v as worst possible result for MAX player then consider every possible move. As these moves
        # backpropagate up a layer, the depth counter is reduced by 1 (terminates when hits zero) and the move we
        # chose is passed to a MIN node. v is keeping track of our current BEST possible play by the MAX player
        # and this should be updated to the maximum of the "old v" and the output of a MIN player's node that our
        # previous MAX player's choice of move m fed into.
        v = float('-inf')
        for m in game.get_legal_moves():
            # Call board evaluation using self.score i.e. custom scoring
            v = max(v, self.min_value(game.forecast_move(m), depth-1))
        return v
        
    def minimax(self, game, depth):
#    """Implement depth-limited minimax search algorithm as described in
#    the lectures.
#
#    This should be a modified version of MINIMAX-DECISION in the AIMA text.
#    https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
#
#    **********************************************************************
#    You MAY add additional methods to this class, or define helper
#          functions to implement the required functionality.
#    **********************************************************************
#
#    Parameters
#    ----------
#    game : isolation.Board
#        An instance of the Isolation game `Board` class representing the
#        current game state
#
#    depth : int
#        Depth is an integer representing the maximum number of plies to
#        search in the game tree before aborting
#
#    Returns
#    -------
#    (int, int)
#        The board coordinates of the best move found in the current search;
#        (-1, -1) if there are no legal moves
#
#    Notes
#    -----
#        (1) You MUST use the `self.score()` method for board evaluation
#            to pass the project tests; you cannot call any other evaluation
#            function directly.
#
#        (2) If you use any helper functions (e.g., as shown in the AIMA
#            pseudocode) then you must copy the timer check into the top of
#            each helper function or else your agent will timeout during
#            testing.
#            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()
        best_score = float('-inf')
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
        else:
            best_move = (-1, -1)
        
        for m in legal_moves:
            # Because we increment the depth below, we are instructing our agent to pick the move that will give our opponent
            # (the MIN player) the highest score (worst for them and best for us). Assuming optimal play by both players this 
            # will then filter down the tree by the recurrent nature of our min and max helper functions (defined above) and 
            # ultimately give us the best possible score i.e. route/moves through the tree
            score = self.min_value(game.forecast_move(m), depth-1)
            if score > best_score:
                best_score = score
                best_move = m
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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
        # Initialize the best move so that this function returns something
        best_move = (-1, -1)
        # IMPORTANT - iterative deepening means we want the depth to increment until time runs out. Our class initialises
        # the search depth to 3 but due to the exponential time complexity (see lecture video), the penalty for exploring 
        # deeper is just multiplication by a scalar and so we should continue to search further until timeout. Thus we 
        # can implement iterative deepening by incrementing the search depth everytime a full search is completed without
        # timeout
        depth = self.search_depth
        # Only continue whilst time on the clock
        while self.time_left() > self.TIMER_THRESHOLD:
            try:
                best_move = self.alphabeta(game, depth)
                depth += 1

            except SearchTimeout:
                # Here we return the existing best move rather than breaking the loop with a "break" command as this will
                # cause timeouts when running tournament.py
                return best_move
        # Return the best move from the last completed search iteration
        return best_move

        
    def ab_max_value(self, game, depth, alpha, beta):
        # Check for timeout in every call to helper functions 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
          
        # If no available moves then return the terminal utility/score
        if not game.get_legal_moves():
            return self.score(game, self)
                
        # Once minimax has been recursively applied enough times the depth countdown hits zero and we return 
        # the utility/score at that particular depth limitation to then be propagated back up the tree
        if depth == 0:
            return self.score(game, self)
                
        # Initially set v as worst possible result for MAX player then consider every possible move. As these moves
        # backpropagate up a layer, the depth counter is reduced by 1 (terminates when hits zero) and the move we
        # chose is passed to a MIN node. v is keeping track of our current BEST possible play by the MAX player
        # and this should be updated to the maximum of the "old v" and the output of a MIN player's node that our
        # previous MAX player's choice of move m fed into.
        v = float('-inf') 
        for m in game.get_legal_moves():
            # Call board evaluation using self.score i.e. custom scoring
            v = max(v, self.ab_min_value(game.forecast_move(m), depth-1, alpha, beta))
            # The alpha-beta bit is to check every possible move to see if it is greater than beta (the current
            # worst performance for a MIN player). If it is then we should pick this as our choice for the MAX 
            # player. We then want to update alpha to be the maximum of this v and the old value of alpha as this
            # now represents a guaranteed score that MAX can achieve and thus is our current estimate of the MAX
            # player's worst possible performance (assuming no better moves lie ahead further down the tree).
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def ab_min_value(self, game, depth, alpha, beta):
        # Check for timeout in every call to helper functions 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
          
        # If no available moves then return the terminal utility/score
        if not game.get_legal_moves():
            return self.score(game, self)
                
        # Once minimax has been recursively applied enough times the depth countdown hits zero and we return 
        # the utility/score at that particular depth limitation to then be propagated back up the tree
        if depth == 0:
            return self.score(game, self)
                
        # Initially set v as worst possible result for MIN player then consider every possible move. As these moves
        # backpropagate up a layer, the depth counter is reduced by 1 (terminates when hits zero) and the move we
        # chose is passed to a MAX node. v is keeping track of our current BEST possible play by the MIN player
        # and this should be updated to the maximum of the "old v" and the output of a MAX player's node that our
        # previous MIN player's choice of move m fed into.
        v = float('inf') 
        for m in game.get_legal_moves():
            # Call board evaluation using self.score i.e. custom scoring
            v = min(v, self.ab_max_value(game.forecast_move(m), depth-1, alpha, beta))
            # The alpha-beta bit is to check every possible move to see if it is less than alpha (the current
            # worst performance for a MAX player). If it is then we should pick this as our choice for the MIN 
            # player. We then want to update beta to be the maximum of this v and the old value of beta as this
            # now represents a guaranteed score that MIN can achieve and thus is our current estimate of the MIN
            # player's worst possible performance (assuming no better moves lie ahead further down the tree).
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def alphabeta(self, game, depth, alpha = float('-inf'), beta = float('inf')):
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()
        # v is initialised to -infinity meaning at least one of the available moves will beat it.
        best_score = float('-inf')
        
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
        else:
            best_move = (-1, -1)
            
        for m in legal_moves:
            # Because we increment the depth below, we are instructing our agent to pick the move that will give our opponent
            # (the MIN player) the highest score (worst for them and best for us). Assuming optimal play by both players this 
            # will then filter down the tree by the recurrent nature of our min and max helper functions (defined above) and 
            # ultimately give us the best possible score i.e. route/moves through the tree
            score = self.ab_min_value(game.forecast_move(m), depth-1, alpha, beta)
            # Everytime we calculate a new score we need to update alpha (worst possible score for a MAX player) in case 
            # it has changed.
            alpha = max(alpha, score)
            if score > best_score:
                best_score = score
                best_move = m
        return best_move


# In[ ]:



