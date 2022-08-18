def calculate_return_of_bet(result, bet, payment_home, payment_draw, payment_away):
    """
    Calculates the return of a bet for a particular match

    Parameters
    ----------
    result: str
        Actual result of the match. Must be one of the following
            - "H"
            - "D"
            - "A"
    bet: str
        Betted result for the match. Must be one of the following
            - "H"
            - "D"
            - "A"
    payment_home: float
        Amount a bettor would receive if a home win was betted and that was what happened
    payment_draw: float
        Amount a bettor would receive if a draw was betted and that was what happened
    payment_away: float
        Amount a bettor would receive if an away win was betted and that was what happened

    Returns
    -------
    bet_return: float
        Amount a bettor would receive, given the bet and the actual result
    """
    
    if result == bet:
        if result == 'H':
            bet_return = payment_home
        elif result == 'D':
            bet_return = payment_draw
        else:
            bet_return = payment_away
    else:
        bet_return = 0
    return float(bet_return)