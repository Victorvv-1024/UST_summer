"""
Functions for dataframe creation
"""

import pickle
import pandas as pd
import numpy as np


from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm
tqdm.pandas()


"""
Auxilary functions
"""
def create_name_id_dict(dataframe):
    """
    A function that is used to create id:name pair
    Args:
        dataframe (pd.dataframe): the dataframe to be examined

    Returns:
        dict: a dictionary of id:name pairs
    """
    name_id = {}
    for i, row in dataframe.iterrows():
        key = row.id
        value = row.author

        key_value_pair = {key:value}

        name_id.update(key_value_pair)
    return name_id

def create_post_comment_name_id(post_dataframe, comment_dataframe):
    post_name_id = {}
    comment_name_id = {}
    post_dataframe['merged'] = post_dataframe.apply(lambda x: [x['author'],x['created_utc']], axis=1)
    comment_dataframe['merged'] = comment_dataframe.apply(lambda x: [x['author'],x['created_utc']], axis = 1)
    for index, row in post_dataframe.iterrows():
        post_name_id.update({row.id:row.merged})
    
    for index, row in comment_dataframe.iterrows():
        comment_name_id.update({row.id:row.merged})
    
    return post_name_id, comment_name_id

"""
Create the daily sub-dataframes from a month dataframe
"""
def workout_time(year, month):
    """
    A function that works out the timestamps of each day in a month
    Args:
        year (int): the year
        month (int): the month

    Returns:
        day_list: the timestamp of each day in a month
    """
    max_day = 31
    month_list = [4, 6, 9, 11]

    if year == 2020 and month == 2:
        max_day = 29
    elif month == 2:
        max_day = 28
    elif month in month_list:
        max_day = 30

    start_date = datetime(year, month, 1, 0) # always start at the first day on each month

    day_list = []

    for i in range(max_day):
        count_time = start_date + relativedelta(days=i)
        count_time = count_time.timestamp()
        day_list.append(count_time)

    return day_list

def create_subframe(year, month, dataframe):
    """
    A funtion that creates sub_dataframes from the parent dataframe based on number of days
    Args:
        year (int): the year
        month (int): the month
        dataframe(pandas.dataframe): the MONTH dataframe we are examine

    Returns:
        df_list: a list of DAY dataframe
    """
    df_list = []

    day_list = workout_time(year, month)

    for i in range(len(day_list)-1):
        left = day_list[i]
        right = day_list[i+1]

        day_df = dataframe[(dataframe.created_utc >= left) & (dataframe.created_utc < right)]

        df_list.append(day_df)

    return df_list

"""
Create the t3-dataframe
"""
def create_t3_df(dataframe):
    """
    A function that creates a t3-dataframe that includes all the authors having parent_id == link_id
    Args:
        dataframe (): the dataframe to be examined

    Returns:
        dataframe: a dataframe of all post authors
    """
    
    comments = dataframe[dataframe['parent_id'] == dataframe['link_id']]
    level_1_comments = comments[comments['parent_id'] != 0]
    level_1_comments['merged'] = level_1_comments.apply(lambda x: [x['author'],x['created_utc']], axis=1) #merge author and created_utc into dict
    t3_df  = level_1_comments.groupby('parent_id')['merged'].apply(list).reset_index(name='merged')
    t3_df['sub_comments'] = t3_df['merged'].progress_apply(lambda x:len(x))
    t3_df.sort_values('sub_comments', inplace=True, ascending=False)
    
    
    return t3_df

"""
Create the t1-dataframe
"""
def create_t1_df(dataframe):
    """
    A function that creates the complement of the t3-df from the same dataframe
    Args:
        dataframe (): the dataframe to be examined

    Returns:
        dataframe: a dataframe of all post authors
    """
    level_1_comments = dataframe[dataframe['parent_id'] == dataframe['link_id']]

    sub_comments = dataframe[~dataframe.index.isin(level_1_comments.index)] #setting the sub comments being those with index that are not in level_1_comments
    sub_comments['merged'] = sub_comments.apply(lambda x: [x['author'],x['created_utc']], axis=1) #merge author and created_utc into dict
    t1_df  = sub_comments.groupby('parent_id')['merged'].apply(list).reset_index(name='merged')
    t1_df['sub_comments'] = t1_df['merged'].progress_apply(lambda x:len(x))
    t1_df.sort_values('sub_comments', inplace=True, ascending=False)
    
    return t1_df

"""
Create the complete dataframe that includes an attribute that analyzes if it is a main post
"""
def is_main(dataframe, t1_name_id, t3_name_id):
    """
    A function that creates an attribute is_main to check if the post is a main post
    Args:
        dataframe (): the dataframe to be examined
        t1_name_id (dict): the id:name pair from t1_df
        t3_name_id (dict): the id:name pair from t3_df

    Returns:
        dataframe: the dataframe with the added attr
    """
    main_list = []
    parent_author_list = []
    parent_utc_list = []
    dataframe = dataframe.reset_index(drop = True)
    
    for index, row in dataframe.iterrows():
        uid = row['parent_id'].split('_')[-1]
        
        pre = row['parent_id'].split('_')[0]
        if pre == 't1':
            main_list.append(0)
            try:
                parent_author_list.append(t1_name_id[uid][0])
                parent_utc_list.append(t1_name_id[uid][1])
            except:
                parent_author_list.append(np.nan)
                parent_utc_list.append(np.nan)
        else:
            main_list.append(1)
            try:
                parent_author_list.append(t3_name_id[uid][0])
                parent_utc_list.append(t3_name_id[uid][1])
            except:
                parent_author_list.append(np.nan)
                parent_utc_list.append(np.nan)

    dataframe['main_post'] =  main_list
    dataframe['parent_author'] = parent_author_list
    dataframe['parent_utc'] = parent_utc_list
    
    return dataframe

def create_complete_df(t1_name_id, t1_df, t3_name_id, t3_df):
    """
    A function that creates the complete dataframe by joining the t1 and t3 dataframe first.
    Then it adds an attribute is_main to the joined dataframe to check if a post is main post
    Args:
        t1_name_id (dict): the id:name pair from t1_df
        t1_df (): t1_dataframe
        t3_name_id (dict): the id:name pair from t3_df
        t3_df (): t3_dataframe

    Returns:
        dataframe: the complete dataframe
    """
    complete_df = pd.concat([t1_df, t3_df])
    complete_df = complete_df.reset_index(drop=True)
    
    complete_df = is_main(complete_df, t1_name_id, t3_name_id)
    
    return complete_df

def create_t3_comment_id_dict(comment_df):
    """
    A function to create id:comment dictionary. 
    Args:
        comment_df (): the comment dataframe

    Returns:
        dict: the key-value pair where key is the id, value is comment content
    """
    comment_id = {}
    comment_df[comment_df['parent_id'] == comment_df['link_id']]
    comment_df['all_comments'] = comment_df.groupby(['parent_id'])['all_comments'].transform(lambda x : ' '.join(x))
    comment_df = comment_df.drop_duplicates()
    for index, row in comment_df.iterrows():
        key = row.id
        value = row.body
        
        comment_id.update({key:value})
        
    return comment_id

def total_post_dict(post_df, comment_id):
    """
    A function to concat main posts with their level 1 comments
    Args:
        post_df (): the post dataframe
        comment_id (dict): he key-value pair where key is the id, value is comment content

    Returns:
        dict: the key-value pair where key is the id of post author, value is the post content + direct comment
    """
    total_dict = {}
    
    for index, row in post_df.iterrows():
        body = row['selftext']
        uid = str(row['parent_id']).split('_')[-1]
        all_comments = comment_id[uid]
        total_string = body + ' ' + all_comments
        
        key_value_pair = {uid:total_string}
        total_dict.update(key_value_pair)
        
    return total_dict
