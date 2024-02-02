import pandas as pd
import datetime

# read data
df = pd.read_csv('./Wimbledon_featured_matches.csv')
print(df.head(), '\n\n')


# change data type
df['elapsed_time'] = pd.to_datetime(df['elapsed_time'], format='%H:%M:%S').dt.time


# find column with none data
missing_values = df.isnull().any()
columns_with_missing_values = missing_values[missing_values].index.tolist()
print("含有空缺值的列:", columns_with_missing_values, '\n\n')


# add p1_winning_streak, p2_winning_streak, add p1_loss_streak, p2_loss_streak, add win_loss_changed
df['p1_winning_streak'] = 0
df['p2_winning_streak'] = 0
df['p1_loss_streak'] = 0
df['p2_loss_streak'] = 0
df['win_loss_changed'] = 0

# init 
p1_winning_streak = 0
p2_winning_streak = 0
p1_loss_streak = 0
p2_loss_streak = 0
victor = df['point_victor'][0]
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_winning_streak = 0
        p2_winning_streak = 0
        p1_loss_streak = 0
        p2_loss_streak = 0
        victor = row['point_victor']
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # win streak and loss streak add data to df
        if row['point_victor'] == 1:
            p1_winning_streak += 1
            p2_winning_streak = 0
            p1_loss_streak = 0
            p2_loss_streak += 1
            df.at[index, 'p1_winning_streak'] = p1_winning_streak
            df.at[index, 'p2_winning_streak'] = p2_winning_streak
            df.at[index, 'p1_loss_streak'] = p1_loss_streak
            df.at[index, 'p2_loss_streak'] = p2_loss_streak
        if row['point_victor'] == 2:
            p1_winning_streak = 0
            p2_winning_streak += 1
            p1_loss_streak += 1
            p2_loss_streak = 0
            df.at[index, 'p1_winning_streak'] = p1_winning_streak
            df.at[index, 'p2_winning_streak'] = p2_winning_streak
            df.at[index, 'p1_loss_streak'] = p1_loss_streak
            df.at[index, 'p2_loss_streak'] = p2_loss_streak

        # win_loss_changed add data to df
        if row['point_victor'] != victor:
            df.at[index, 'win_loss_changed'] = 1
            victor = row['point_victor']


# add serve_ace_times
df['p1_serve_ace_times'] = 0
df['p2_serve_ace_times'] = 0

p1_serve_ace_times = 0
p2_serve_ace_times = 0
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_serve_ace_times = 0
        p2_serve_ace_times = 0
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # serve_ace_times add data to df
        if row['p1_ace'] == 1:
            p1_serve_ace_times += 1
        if row['p2_ace'] == 1:
            p2_serve_ace_times += 1

        df.at[index, 'p2_serve_ace_times'] = p2_serve_ace_times
        df.at[index, 'p1_serve_ace_times'] = p1_serve_ace_times

# add untouchable_shot_times
df['p1_untouchable_shot_times'] = 0
df['p2_untouchable_shot_times'] = 0

# init
p1_untouchable_shot_times = 0
p2_untouchable_shot_times = 0
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_untouchable_shot_times = 0
        p2_untouchable_shot_times = 0
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # untouchable_shot_times add data to df
        if row['p1_winner'] == 1:
            p1_untouchable_shot_times += 1
        if row['p2_winner'] == 1:
            p2_untouchable_shot_times += 1

        df.at[index, 'p1_untouchable_shot_times'] = p1_untouchable_shot_times
        df.at[index, 'p2_untouchable_shot_times'] = p2_untouchable_shot_times


# add net_pt_times
df['p1_net_pt_times'] = 0
df['p2_net_pt_times'] = 0

# init
p1_net_pt_times = 0
p2_net_pt_times = 0
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_net_pt_times = 0
        p2_net_pt_times = 0
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # net_pt_times add data to df
        if row['p1_net_pt'] == 1:
            p1_net_pt_times += 1
        if row['p2_net_pt'] == 1:
            p2_net_pt_times += 1

        df.at[index, 'p1_net_pt_times'] = p1_net_pt_times
        df.at[index, 'p2_net_pt_times'] = p2_net_pt_times


# add net_pt_won_times
df['p1_net_pt_won_times'] = 0
df['p2_net_pt_won_times'] = 0

# init
p1_net_pt_won_times = 0
p2_net_pt_won_times = 0
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_net_pt_won_times = 0
        p2_net_pt_won_times = 0
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # net_pt_won_times add data to df
        if row['p1_net_pt_won'] == 1:
            p1_net_pt_won_times += 1
        if row['p2_net_pt_won'] == 1:
            p2_net_pt_won_times += 1

        df.at[index, 'p1_net_pt_won_times'] = p1_net_pt_won_times
        df.at[index, 'p2_net_pt_won_times'] = p2_net_pt_won_times


# add double_fault_times
df['p1_double_fault_times'] = 0
df['p2_double_fault_times'] = 0

# init
p1_double_fault_times = 0
p2_double_fault_times = 0
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_double_fault_times = 0
        p2_double_fault_times = 0
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # double_fault_times add data to df
        if row['p1_double_fault'] == 1:
            p1_double_fault_times += 1
        if row['p2_double_fault'] == 1:
            p2_double_fault_times += 1

        df.at[index, 'p1_double_fault_times'] = p1_double_fault_times
        df.at[index, 'p2_double_fault_times'] = p2_double_fault_times


# add unforced_error_times
df['p1_unforced_error_times'] = 0
df['p2_unforced_error_times'] = 0

# init
p1_unforced_error_times = 0
p2_unforced_error_times = 0
match_id = df['match_id'][0]

for index, row in df.iterrows():
    # new match
    if row['match_id'] != match_id:
        p1_unforced_error_times = 0
        p2_unforced_error_times = 0
        match_id = row['match_id']
    if row['match_id'] == match_id:
        # unforced_error_times add data to df
        if row['p1_unf_err'] == 1:
            p1_unforced_error_times += 1
        if row['p2_unf_err'] == 1:
            p2_unforced_error_times += 1

        df.at[index, 'p1_unforced_error_times'] = p1_unforced_error_times
        df.at[index, 'p2_unforced_error_times'] = p2_unforced_error_times


# show df
print(df.head(), '\n\n')


# output csv
df.to_csv('./out.csv', index=False)


