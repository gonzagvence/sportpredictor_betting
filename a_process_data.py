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
    if  h <= a:
        expected_outcome = 'H'
    elif a <= h:
        expected_outcome = 'A'
    if -0.3<=(h-a)<=0.3:
        expected_outcome = "D"
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
    if h <= d and h <= a:
        prob_expected_outcome = h
    elif d <= h and d <= a:
        prob_expected_outcome = d
    elif a <= h and a <= d:
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


def masprob(p, r):
    if p==r:
        return "WIN"
    else:
        return "LOSE"
    
def siemprelocal(r):
    if r=="H":
        return "WIN"
    else:
        return "LOSE"
def masde3goles(hg, ag):
    if hg+ag >= 3:
        return "WIN"
    else:
        return "LOSE"



if __name__ == '__main__':

    # Read a csv file
    df = pd.read_csv('ARG.csv', sep=',')
    
    df.loc[:, 'Calculate Expected Outcome'] = np.vectorize(calc_expected_outcome)(df['AvgH'], df['AvgD'], df['AvgA'])
    
    df.loc[:, 'calc_expected_prob'] = np.vectorize(calc_expected_prob)(df['AvgH'], df['AvgD'], df['AvgA'])
    
    df.loc[:, 'difference_between_first_and_second_outcome'] = np.vectorize(difference_between_first_and_second_outcome)(df['AvgH'], df['AvgD'], df['AvgA'])

    df.loc[:, 'difference_between_first_and_third_outcome'] = np.vectorize(difference_between_first_and_third_outcome)(df['AvgH'], df['AvgD'], df['AvgA'])

    df.loc[:, 'MasProb'] = np.vectorize(masprob)(df['Calculate Expected Outcome'], df['Res'])
    df.loc[:, 'SiempreLocal'] = np.vectorize(siemprelocal)( df['Res'])
    df.loc[:, '+de3goles'] = np.vectorize(masde3goles)( df['HG'], df['AG'])
    df_masgoles = df[df['+de3goles'] == 'WIN']
    masgoleslocal = list(df_masgoles['Home'])
    masgolesvis = list(df_masgoles['Away'])

masgoles = masgoleslocal+masgolesvis

import statistics
from statistics import mode
 
def most_common(List):
    return(mode(List))
indicado = most_common(masgoles)
print(indicado)
def process_df(df):
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
    
    
    
    return df

