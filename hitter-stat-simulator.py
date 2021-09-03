import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import random as rm

def handle_batting_transition(start_state, batting_outcome, runs, outs):
    """Takes in current runner-on-base situation, runs, outs, and batting outcome to return end state info

    Args:
        start_state (str): current runner-on-base state, e.g. 'AAA' = bases empty, 'A1A' = runner on 2nd only
        batting_outcome (str): monte carlo output of batting situation
        runs (int): number to track runs scored in current innings
        outs (int): number to track outs in current inning 
    Return:
        
    """
    #start_state = current_state
    #outcome = np.random.choice(outcomes, replace=True, p=transition_matrix[0])
    
    # Outs and home runs
    if batting_outcome == 'O':
        outs = outs + 1
        # No change in state
        current_state = start_state
    
    elif batting_outcome == 'HR':
        # Home run scores and all runners clear bases and score
        
        if start_state == 'AAA':
            runs = runs + 1
        elif start_state == '1AA':
            runs = runs + 2
        elif start_state == 'A1A':
            runs = runs + 2
        elif start_state == 'AA1':
            runs = runs + 2
        elif start_state == '1A1':
            runs = runs + 3
        elif start_state == '11A':
            runs = runs + 3
        elif start_state == 'A11':
            runs = runs + 3
        elif start_state == '111': # Grand Slam
            runs = runs + 4
        # Home run will always reset the current state to AAA (because all runners are scoring runs)
        current_state = 'AAA'
    
    # Hit by pitch, walks, and singles
    elif batting_outcome in ['BB','HBP','1B']:
        # Advance all base runners by 1, add a man to 1st base, runners on 3rd score
        
        if start_state == 'AAA':
            current_state = '1AA'
        elif start_state == '1AA':
            current_state = '11A'
        elif start_state == 'A1A':
            current_state = '1A1'
        elif start_state == 'AA1':
            # Runner on 3rd scores
            runs = runs + 1
            current_state = '1AA'
        elif start_state == '1A1':
            # Runner on 3rd scores
            runs = runs + 1
            current_state = '11A'
        elif start_state == '11A':
            current_state = '111'
        elif start_state == 'A11':
            # Runner on third scores
            runs = runs + 1
            current_state = '1A1'
        elif start_state == '111':
            # Runner on third scores
            runs = runs + 1
            current_state = '111'
    # 2B
    elif batting_outcome == '2B':
        
        # Advance all base runners by 2, add a man to 2nd base, runners on 2nd and 3rd score
        
        if start_state == 'AAA':
            current_state = 'A1A'
        elif start_state == '1AA':
            current_state = 'A11'
        elif start_state == 'A1A':
            # Runner on 2nd scores
            runs = runs + 1
            current_state = 'A1A'
        elif start_state == 'AA1':
            # Runner on 3rd scores
            runs = runs + 1
            current_state = 'A1A'
            
        elif start_state == '1A1':
            # Runner on 3rd scores
            runs = runs + 1
            current_state = 'A11'
        elif start_state == '11A':
            # Runner on 2nd scores
            runs = runs + 1
            current_state = 'A11'
        elif start_state == 'A11':
            # Runner on third scores
            runs = runs + 2
            current_state = 'A1A'
        elif start_state == '111':
            # Runner on third scores
            runs = runs + 2
            current_state = 'A11'
    
    # 3B
    elif batting_outcome == '3B':
        # Advance all base runners by 3, add a man to 3rd base, runners on 1st, 2nd, and 3rd score
        
        if start_state == 'AAA':
            current_state = 'AA1'
        elif start_state == '1AA':
            # Runner on 1st scores
            runs = runs + 1
            current_state = 'AA1'
        elif start_state == 'A1A':
            # Runner on 2nd scores
            runs = runs + 1
            current_state = 'AA1'
        elif start_state == 'AA1':
            # Runner on 3rd scores
            runs = runs + 1
            current_state = 'AA1'
        elif start_state == '1A1':
            # Runners on 1st and 3rd scores
            runs = runs + 2
            current_state = 'AA1'
        elif start_state == '11A':
            # Runners on 1st and 2nd scores
            runs = runs + 2
            current_state = 'AA1'
        elif start_state == 'A11':
            # Runner on third scores
            runs = runs + 2
            current_state = 'AA1'
        elif start_state == '111':
            # Runner on third scores
            runs = runs + 3
            current_state = 'AA1'
    
    # Dictionary of final outputs
    output_dict = {'current_state': current_state,
                   'runs': runs,
                   'outs': outs}

    return(output_dict)

def load_prep_inputs():
    """Preps MLB individual player stats data for monte carlo simulation - currently very crude 
    Args:
        
    Return:
        DataFrame with probabilities for plate appearance events, for all players in current lineup
    """

    # Read in data
    dodgers_df = pd.read_csv('./lineups/dodgers.csv')
    #padres_df = pd.read_csv('/lineups/padres.csv')

    # Temp 
    data = dodgers_df

    # Derive some fields
    data['O'] = data['PA'] - data['H'] - data['BB'] - data['HBP']
    data['1B'] = data['H'] - data['2B'] - data['3B'] - data['HR']

    # Filter to only independent variables of interest
    cols = ['player','Tm','PA','1B','2B','3B','HR','BB','HBP','O']
    data = data[cols]

    # Filter to teams of interest
    #data = data[data.Tm.isin(teams)]

    # Convert fields of interest to probabilities, based on season-to-date data
    data['1B'] = data['1B'] / data['PA']  
    data['2B'] = data['2B'] / data['PA']  
    data['3B'] = data['3B'] / data['PA']
    data['HR'] = data['HR'] / data['PA']
    data['BB'] = data['BB'] / data['PA']
    data['HBP'] = data['HBP'] / data['PA']
    data['O'] = data['O'] / data['PA']

    # Drop some cols before returning
    data = data.drop(['Tm','PA'], axis=1).reset_index(drop=True)

    return(data)

def simulate_inning(current_batting_order, player_prob_inputs):
    """Perform monte carlo simulation for a hypothetical inning of baseball

    Args:
        data: DataFrame of probabilities that will be used to drive events in simulation
    Return:

    """
    # Begin simulation from start of inning and whichever the current batter is (1st inning will always be 1st batter)
    outcomes = ['1B','2B','3B','HR','BB','HBP','O']
    # Convert thresholds df to a matrix for np.random.choice()

    # Convert dataframe to numpy array    
    player_probs_matrix = player_prob_inputs.drop('player', axis=1).to_numpy()
    
    # Dictionary that will contain lists with each batter's outcomes
    player_stats_dict = {player_name:[] for player_name in player_prob_inputs.iloc[:,0]}

    # Initial inputs, always start a new inning like this
    current_state = 'AAA'
    runs = 0
    outs = 0

    # Iterate through batting outcomes until 3 outs occur
    while outs < 3:

        # Capture the current batter name for box score
        current_batter_name = player_prob_inputs.iloc[current_batting_order].iloc[0]

        # Evaluate the outcome of the current batter's hitting situation
        #batting_outcome = np.random.choice(outcomes, replace=True, p=transition_matrix[current_batting_order])
        batting_outcome = np.random.choice(outcomes, replace=True, p=player_probs_matrix[current_batting_order])

        # Track player stats
        player_stats_dict[current_batter_name].append(batting_outcome)

        # Transition to next state, if necessary, based on hitting outcome
        batting_outputs = handle_batting_transition(current_state, batting_outcome, runs, outs)

        # Return the runs and outs from the inning, for now - this is going to be much more robust later on
        current_state = batting_outputs.get('current_state')
        runs = batting_outputs.get('runs')
        outs = batting_outputs.get('outs')

        # Increment batting order - jump back to first batter if the last batter just went (batter 8)
        current_batting_order = current_batting_order + 1 if current_batting_order < 8 else 0

        # Debug output
        #print('******************************************')
        #print('Current Event: ', current_batter_name, ' ', batting_outcome)
        #print('Current State: ', current_state)
        #print('Runs: ', runs)
        #print('Outs: ', outs)
        #print('******************************************')
    
    output_dict = {'current_batting_order': current_batting_order,
                   'current_state': current_state,
                   'runs': runs,
                   'outs': outs,
                   'player_stats': player_stats_dict}

    return(output_dict)


if __name__ == "__main__":

    # Load & prep input data
    player_prob_inputs = load_prep_inputs()
    # This will store each innings runs
    total_runs_scored = []

    # Number of simulations to run
    num_sims = 100

    # Simulate a 9 inning game
    for i in range(num_sims):
        
        # Initialize 1st inning
        current_batting_order = 0
        innings = 1
        game_runs_scored = []

        # Create a dictionary that will hold each players game stats
        player_stats_dict = {player_name:[] for player_name in player_prob_inputs.iloc[:,0]}
        
        # Grabbing just the player_names also because using a dictionary will throw them out of order
        player_names =  [player_name for player_name in player_prob_inputs.iloc[:,0]]

        while innings <= 9:
            # Simulate current inning
            inning_outputs = simulate_inning(current_batting_order, player_prob_inputs)

            # Capture inning outputs
            inning_runs = inning_outputs.get('runs')
            inning_outs = inning_outputs.get('outs')
            current_state = inning_outputs.get('current_state')
            current_batting_order = inning_outputs.get('current_batting_order')


            # Append the current inning's player stats to the game dict
            #player_stats_dict = { key:player_stats_dict.get(key,[])+inning_outputs.get('player_stats').get(key,[]) for key in set(list(player_stats_dict.keys())+list(inning_outputs.get('player_stats').keys())) }
            player_stats_dict = { key:player_stats_dict.get(key,[])+inning_outputs.get('player_stats').get(key,[]) for key in player_names}
            
            # Tracking full game runs scored
            game_runs_scored.append(inning_runs)

            # Increment innings
            innings = innings + 1

        # Append the game's total runs 
        total_runs_scored.append(pd.Series(game_runs_scored).sum())

        # Display player stats
        print('***************************************')
        print('PLAYER STATS:  \n')
        print(player_stats_dict)
        print('TOTAL RUNS: \t', pd.Series(game_runs_scored).sum())
        print('***************************************')


    # Final output for now is just a dataframe with the sim results - sim_id and runs_scored
    df = pd.DataFrame(data={'sim_id':range(1, num_sims+1), 
                            'runs': total_runs_scored})



    print('***************************************')
    print('SIMULATION RESULTS: \n')
    print(df)
    print('***************************************')
