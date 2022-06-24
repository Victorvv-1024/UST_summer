"""
This file is used to clean the dataframe.
It firstly cleans the dataframe by author names in 3 approaches.
1. It detects if the author name includes 'bot' in it. If both 'not'and 'bot' are included in the name, it is not a BOT.
2. It detects if the author name is one of the bot list given by Reddit
3. It detects if the author name is one of the popular bot post author
It therefore cleans the dataframe by the parent id. If the author replies to a post which is classified as BOT, this author's data is neglected.
"""

import re
import pandas as pd

from .dataframe_utils import create_name_id_dict


def advanced_clean(name):
    """
    A function that cleans the author name
    Args:
        name (string): the author name

    Returns:
        bool: True if it is a bot, else False
    """
    
    tokens = re.findall(r'[A-Z](?:[A-Z]*(?![a-z])|[a-z]*)', name)
    tokens = list(map(lambda x: x.lower(), tokens))
    
    tokens = list (map(lambda x: re.split('[^a-zA-Z]',x), tokens))
    tokens = [item for sublist in tokens for item in sublist]
    
    if 'not' in tokens and 'bot' in tokens:
        return False

    elif 'bot' in tokens:
        return True
    
    return False

def clean_by_parent(dataframe, name_id):
    """
    A function that is used to clean any identified BOT from the parent_id in a dataframe
    Args:
        dataframe (pandas.dataframe): the dataframe to be exmained
        name_id (dictionary): a dictionary having the id as the key, its corresponding name as value

    Returns:
        dataframe: the dataframe that is cleaned by parent_id
    """
    bot = []
    for index, row in dataframe.iterrows():
        uid = str(row['parent_id']).split('_')[-1]
        # use the dictionary to find the corresponding author name
        try:
            author_name = name_id[uid]
        except:
            # this would happen if the poster created the post in posterior months
            continue
        if advanced_clean(author_name):
            bot.append(author_name)
    
    dataframe['is_bot'] = dataframe.author.isin(bot)
    bot = dataframe[dataframe.is_bot == True]
    dataframe = dataframe[dataframe.is_bot == False]
    
    return bot, dataframe

def full_clean (dataframe, bot_list, popular_bot, by_parent=False):
    """
    A function that fully cleans the dataframe
    Args:
        dataframe (): the dataframe to be cleaned
        bot_list (): the dataframe of the bot names given by Reddit
        popular_bot (): the dataframe of the popular author name that is bot from Reddit
        by_parent (bool, option default = False): if True it cleans the datframe by parent_id

    Returns:
        dataframe: a dataframe that includes all the found bots and a cleaned dataframe
    """
    # shallow clean
    dataframe = dataframe[dataframe.author != '[deleted]']
    dataframe = dataframe[dataframe.author != 'AutoModerator']
    dataframe = dataframe[dataframe.author != 'VisualMod']
    
    # deep clean 
    dataframe['is_bot']  = dataframe['author'].progress_apply(advanced_clean)
    bot_df1 = dataframe[dataframe.is_bot == True]
    dataframe = dataframe[dataframe.is_bot == False]
    
    # further clean
    dataframe['is_bot'] = dataframe.author.isin(bot_list.Username)
    bot_df2 = dataframe[dataframe.is_bot == True]
    dataframe = dataframe[dataframe.is_bot == False]
    
    dataframe['is_bot'] = dataframe.author.isin(popular_bot.author)
    bot_df3 = dataframe[dataframe.is_bot == True]
    dataframe = dataframe[dataframe.is_bot == False]
    bot = pd.concat([bot_df1, bot_df2,bot_df3])

    # clean by parent id
    if by_parent:
        bot_df4, dataframe = clean_by_parent(dataframe, create_name_id_dict(dataframe))
        bot = pd.concat([bot, bot_df4])
    
    bot = bot.drop_duplicates()
    
    return bot, dataframe.reset_index(drop=True)