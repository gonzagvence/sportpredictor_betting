import pandas as pd
import numpy as np



def calc_expected_outcome(h, d, a):
    """
    Calculate which event (home win, draw, away win) is the most likely, according to implied probabilities

    Parameters
    ----------
    h: float
        Expected probability of a home win
    d: float
        Expected probability of a draw
    a: float
        Expected probability of a away win

    Returns
    -------
    expected_outcome: str
        String indicating if the most expected outcome is a home win ('H'), draw ('D') or away win ('A')

    """
    # Decide the expected outcome, evaluating probabilities
    if h >= d and h >= a:
        expected_outcome = 'H'
    elif d >= h and d >= a:
        expected_outcome = 'D'
    elif a >= h and a >= d:
        expected_outcome = 'A'
    return expected_outcome



def calc_expected_prob(h, d, a):
    """
    Calculate the probability of the event (home win, draw, away win) that is most likely,
    according to implied probabilities

    Parameters
    ----------
    h: float
        Expected probability of a home win
    d: float
        Expected probability of a draw
    a: float
        Expected probability of a away win

    Returns
    -------
    prob_expected_outcome: float
        Probability of the most expected outcome

    """
    # Decide the probability expected outcome, evaluating probabilities
    if h >= d and h >= a:
        prob_expected_outcome = h
    elif d >= h and d >= a:
        prob_expected_outcome = d
    elif a >= h and a >= d:
        prob_expected_outcome = a
    return prob_expected_outcome


def difference_between_first_and_second_outcome(h, d, a):
    """
    Calculates difference between the most likely and the second most likely outcome

    Parameters
    ----------
    h: float
        Expected probability of a home win
    d: float
        Expected probability of a draw
    a: float
        Expected probability of a away win

    Returns
    -------
    difference: float
        Difference between the most likely and the second most likely outcome
    """
    probs = [h, d, a]
    probs.sort()
    difference = probs[2] - probs[1]
    return difference

def difference_between_first_and_third_outcome(h, d, a):
    """
    Calculates difference between the most likely and the third most likely outcome

    Parameters
    ----------
    h: float
        Expected probability of a home win
    d: float
        Expected probability of a draw
    a: float
        Expected probability of a away win

    Returns
    -------
    difference: float
        Difference between the most likely and the third most likely outcome
    """
    probs = [h, d, a]
    probs.sort()
    difference = probs[2] - probs[0]
    return difference

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

def estrategia(r,h,d,a):
        if r > 0.92:
            return "D"
        else:
            return calc_expected_outcome(h, d, a)
if __name__ == '__main__':

    # Read a csv file
    df = pd.read_csv('ARG.csv', sep=',')
    PL = pd.read_csv('Jugadores Premier League_csv.csv', sep=',')
    


def dfhola(df):
    """
    In this function, we process the downloaded dataframe, creating the following columns
        - Year
        - Expected probability of each outcome (Home Win, Draw, Away Win)
        - Most expected outcome
        - Probability of most expected outcome
        - Level of risk of a matches' bets (as 1 - standard deviation of probabilities)
        - Calculate difference between the most likely outcome and the second most likely outcome
        - Calculate difference between the most likely outcome and the third most likely outcome (i.e. the least likely)

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with historical matches information

    Returns
    -------
    df: pd.DataFrame
        DataFrame with historical matches information, with new columns
    """

    # Make Year column
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
    df.loc[:, 'Year'] = df['Date'].dt.year

    # As AvgH, represents the pay that you would get if the home team wins, we calculate the raw expected probability of
    # the home team winning as 1/AvgH. However, the sum of these probabilities don't add up to 1.
    # Therefore, we scale them to get to this

    # We first calculate raw probabilities
    df.loc[:, 'ProbH'] = 1 / df['AvgH']
    df.loc[:, 'ProbD'] = 1 / df['AvgD']
    df.loc[:, 'ProbA'] = 1 / df['AvgA']

    # We calculate the sum, used to scale raw probabilities
    df.loc[:, 'Sum_Probs'] = df['ProbH'] + df['ProbD'] + df['ProbA']
    df.loc[:, 'ProbH'] = df['ProbH'] / df['Sum_Probs']
    df.loc[:, 'ProbD'] = df['ProbD'] / df['Sum_Probs']
    df.loc[:, 'ProbA'] = df['ProbA'] / df['Sum_Probs']

    # Calculation of most expected outcome and the probability of the most expected outcome
    df.loc[:, 'Expected Outcome'] = np.vectorize(calc_expected_outcome)(df['ProbH'], df['ProbD'],
                                                                        df['ProbA'])

    df.loc[:, 'Expected Prob'] = np.vectorize(calc_expected_prob)(df['ProbH'], df['ProbD'],
                                                                  df['ProbA'])

    # Calculate risk level of a matches' bet (with standard deviation)
    df.loc[:, 'RiskLevel'] = 1 - df[['ProbH', 'ProbD', 'ProbA']].std(axis=1)

    # Calculate difference between probability outcomes
    df.loc[:, 'DiffFirstSecond'] = np.vectorize(difference_between_first_and_second_outcome)(df['ProbH'], df['ProbD'],
                                                                                             df['ProbA'])
    df.loc[:, 'DiffFirstThird'] = np.vectorize(difference_between_first_and_third_outcome)(df['ProbH'], df['ProbD'],
                                                                                           df['ProbA'])
    df.loc[:, 'estrategia_infalible'] = np.vectorize(calculate_return_of_bet)(df['Res'], df['Res'], df['AvgH'], df['AvgD'], df['AvgA'])

    df.loc[:, 'estrategia'] = np.vectorize(estrategia)(df['RiskLevel'], df['ProbH'], df['ProbD'], df['ProbA'])
    df.loc[:, 'AE'] = np.vectorize(calculate_return_of_bet)(df['Res'], df['estrategia'], df['AvgH'], df['AvgD'], df['AvgA'])
    df.loc[:, 'AHW'] = np.vectorize(calculate_return_of_bet)(df['Res'], "H", df['AvgH'], df['AvgD'], df['AvgA'])
    df.loc[:, 'AAW'] = np.vectorize(calculate_return_of_bet)(df['Res'], "A", df['AvgH'], df['AvgD'], df['AvgA'])
    df.loc[:, 'AEXP'] = np.vectorize(calculate_return_of_bet)(df['Res'], df['Expected Outcome'], df['AvgH'], df['AvgD'], df['AvgA'])

    
    
    return df


    

dfhola(df)


estani = pd.DataFrame()
estani.loc[:, 'Date'] = pd.to_datetime(df['Date'])
estani.loc[:, 'Home'] = (df['Home'])
estani.loc[:, 'Away'] = (df['Away'])
estani.loc[:, 'Bet'] = (df['estrategia'])
estani.to_csv('The_best_bet.csv')

def puntosloc(r):
    if r == "H":
        return 3
    elif r == "D":
        return 1
    else:
        return 0
def puntosvis(r):
    if r == "A":
        return 3
    elif r == "D":
        return 1
    else:
        return 0

#Independiente
IndependienteLocal = df[df['Home'] == 'Independiente']
IndependienteLocal.loc[:, 'P'] = np.vectorize(puntosloc)(IndependienteLocal['Res'])
IndependienteVis = df[df['Away'] == 'Independiente']
IndependienteVis.loc[:, 'P'] = np.vectorize(puntosvis)(IndependienteVis['Res'])
golestotalesind = sum(list(IndependienteLocal["HG"]))+sum(list(IndependienteVis["AG"]))
golestotalesrind = sum(list(IndependienteLocal["AG"]))+sum(list(IndependienteVis["HG"]))
puntostotalesind = sum(list(IndependienteLocal["P"]))+sum(list(IndependienteVis["P"]))

#Boca
BocaLocal = df[df['Home'] == 'Boca Juniors']
BocaLocal.loc[:, 'P'] = np.vectorize(puntosloc)(BocaLocal['Res'])
BocaVis = df[df['Away'] == 'Boca Juniors']
BocaVis.loc[:, 'P'] = np.vectorize(puntosvis)(BocaVis['Res'])
golestotalesboca = sum(list(BocaLocal["HG"]))+sum(list(BocaVis["AG"]))
golestotalesrboca = sum(list(BocaLocal["AG"]))+sum(list(BocaVis["HG"]))
puntostotalesboca = sum(list(BocaLocal["P"]))+sum(list(BocaVis["P"]))

#River Plate
RiverPlateLocal = df[df['Home'] == 'River Plate']
RiverPlateLocal.loc[:, 'P'] = np.vectorize(puntosloc)(RiverPlateLocal['Res'])
RiverPlateVis = df[df['Away'] == 'River Plate']
RiverPlateVis.loc[:, 'P'] = np.vectorize(puntosvis)(RiverPlateVis['Res'])
golestotalesriv = sum(list(RiverPlateLocal["HG"]))+sum(list(RiverPlateVis["AG"]))
golestotalesrriv = sum(list(RiverPlateLocal["AG"]))+sum(list(RiverPlateVis["HG"]))
puntostotalesriv = sum(list(RiverPlateLocal["P"]))+sum(list(RiverPlateVis["P"]))

#San Lorenzo
SanLorenzoLocal = df[df['Home'] == 'San Lorenzo']
SanLorenzoLocal.loc[:, 'P'] = np.vectorize(puntosloc)(SanLorenzoLocal['Res'])
SanLorenzoVis = df[df['Away'] == 'San Lorenzo']
SanLorenzoVis.loc[:, 'P'] = np.vectorize(puntosvis)(SanLorenzoVis['Res'])
golestotalessl = sum(list(SanLorenzoLocal["HG"]))+sum(list(SanLorenzoVis["AG"]))
golestotalesrsl = sum(list(SanLorenzoLocal["AG"]))+sum(list(SanLorenzoVis["HG"]))
puntostotalessl = sum(list(SanLorenzoLocal["P"]))+sum(list(SanLorenzoVis["P"]))

#Racing
RacingLocal = df[df['Home'] == 'Racing Club']
RacingLocal.loc[:, 'P'] = np.vectorize(puntosloc)(RacingLocal['Res'])
RacingVis = df[df['Away'] == 'Racing Club']
RacingVis.loc[:, 'P'] = np.vectorize(puntosvis)(RacingVis['Res'])
golestotalesrc = sum(list(RacingLocal["HG"]))+sum(list(RacingVis["AG"]))
golestotalesrrc = sum(list(RacingLocal["AG"]))+sum(list(RacingVis["HG"]))
puntostotalesrc = sum(list(RacingLocal["P"]))+sum(list(RacingVis["P"]))

#Equipos
Equipos = ["Boca Juniors","River Plate","Independiente", "Racing Club", "San Lorenzo"]
Goles = [golestotalesboca,golestotalesriv,golestotalesind,golestotalesrc,golestotalessl]
GolesRec = [golestotalesrboca,golestotalesrriv,golestotalesrind,golestotalesrrc,golestotalesrsl]
Puntos = [puntostotalesboca,puntostotalesriv,puntostotalesind,puntostotalesrc,puntostotalessl]
CincoGrandes = pd.DataFrame()
CincoGrandes.loc[:, 'Equipos'] = Equipos
CincoGrandes.loc[:, 'Goles Totales'] = Goles
CincoGrandes.loc[:, 'Goles Recibidos Totales'] = GolesRec
CincoGrandes.loc[:, 'Puntos Totales'] = Puntos
CincoGrandes.to_csv('cincograndes.csv')

#Intento