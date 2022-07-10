"""
Functions to get the information
"""


import pandas as pd
import numpy as np
import networkx as nx


from tqdm.notebook import tqdm
tqdm.pandas()

"""
Create edge list and fetch edge list-related information
"""
def merge_df(comment_df, post_df):
    """
    A function to merge the comment df and post df. Because post dataframe does not have parent_id and link_id
    Args:
        comment_df (): comment dataframe
        post_df (): post dataframe

    Returns:
        dataframe: the merged dataframe
    """
    col_zero = np.zeros(post_df.shape[0])
    
    post_df['parent_id'] = pd.Series(col_zero)
    post_df['link_id'] = pd.Series(col_zero)
    concat_df = pd.concat([comment_df, post_df])
    concat_df = concat_df.reset_index(drop=True)
    return concat_df

def get_parallel_edges(comment_df, post_df, startdate):
    """
    A function that returns the parallel edge list dataframe that has 3 attributes Source, Target and is_main
    Args:
        comment_df (): the comment dataframe
        post_df (): the post dataframe
    Returns:
        dataframe: a dataframe of edge list
    """
    startdate = startdate.timestamp()
    older_post_list = []
    source_list = []
    target_list = []
    is_main_list = []
    source_utc_list = []
    target_utc_list = []
    non_singleton = set()
    potential_singleton = set()
    non_single_post = set()
    
    authordf = set(post_df[post_df['parent_id']==0]['author'])
    for index, row in comment_df.iterrows():
        target = row['parent_author']
        non_single_post.add(target)
        author_list = row['merged']
        post_type = row['parent_id'].split('_')[0]
        target_utc = row['parent_utc']
        if post_type == 't1':
            main_post = 0
        else:
            main_post = 1
        if pd.isna(target) == False:
            non_singleton.add(target)
            for author in author_list:
                source_list.append(author[0])
                target_list.append(target)
                source_utc_list.append(author[1])
                target_utc_list.append(target_utc)
                non_singleton.add(author[0])
                is_main_list.append(main_post)
                if author[1] > startdate:
                    older_post_list.append(0)
                else:
                    older_post_list.append(1)
        else:
            for author in author_list:
                potential_singleton.add(author[0])
    
    single_main = authordf - non_single_post
    singletons = set(potential_singleton) - set(non_singleton)
    singletons = singletons.union(single_main)
    for singleton in singletons:
        source_list.append(singleton)
        target_list.append(singleton)
        is_main_list.append(np.nan)
        source_utc_list.append(np.nan)
        target_utc_list.append(np.nan)
        older_post_list.append(np.nan)
    edge_list = pd.DataFrame(list(zip(source_list, target_list, is_main_list, source_utc_list, target_utc_list, older_post_list)), columns = ['Source', 'Target', 'is_main', 'source_utc', 'target_utc', 'main_is_old'])
    
    return edge_list

def get_weighted_edges(parallel_edges):
    """
    A function that returns the weighted edge dataframe that has 3 attributes Source, Target and weights
    Args:
        parallel_edges (dataframe): the parallel edge list dataframe

    Returns:
        dataframe: the weighted edges
    """
    weighted_edges = parallel_edges[['Source', 'Target']]
    weighted_edges = weighted_edges.groupby(weighted_edges.columns.tolist()).size().reset_index().rename(columns={0:'weight'})

    return weighted_edges

def get_edge_info(weighted_edges, post_authors):
    """
    A function that returns edge list related information as a dataframe
    Args:
        weighted_edges (dataframe): the weighed edge list dataframe
        post_authors (list): a list of all post author names
    
    Returns:
        dataframe: the information dataframe
    """
    #Returns a graph from Pandas DataFrame containing an edge list.
    G = nx.from_pandas_edgelist(weighted_edges, source = "Source", target = "Target", edge_attr= ["weight", "inverse_weight"], create_using = nx.DiGraph())
    
    # these two attr only for post nodes
    avg_neighbour_deg = nx.average_neighbor_degree(G, weight="weight", nodes=post_authors)
    avg_neighbour_deg = pd.DataFrame(avg_neighbour_deg.items(), columns=['author', 'avg_neighbout_degree'])
    
    cluster_coeff = nx.clustering(G, nodes=post_authors, weight='weight')
    cluster_coeff = pd.DataFrame(cluster_coeff.items(), columns=['author', 'cluster_coeff'])
    
    post_node_merged_df = avg_neighbour_deg.merge(cluster_coeff)
    
    # these three attr for all nodes
    in_deg_cent = nx.in_degree_centrality(G)
    in_deg_cent = pd.DataFrame(in_deg_cent.items(), columns=['author', 'in_deg_cent'])
    
    out_deg_cent = nx.out_degree_centrality(G)
    out_deg_cent = pd.DataFrame(out_deg_cent.items(), columns=['author', 'out_deg_cent'])
    
    eigenvec_cent = nx.eigenvector_centrality(G, max_iter = 500, weight='weight')
    eigenvec_cent = pd.DataFrame(eigenvec_cent.items(), columns=['author', 'eigenvec_cent'])
    
    all_node_merged_df = in_deg_cent.merge(out_deg_cent)
    all_node_merged_df = all_node_merged_df.merge(eigenvec_cent)
    all_node_merged_df = all_node_merged_df.merge(post_node_merged_df, on=['author'], how='left')
    
    # these attr is done within for-loop for every post node
    ego_graph_density_dict = dict()
    closeness_cent_dict = dict()
    D = G.reverse()
    for node in post_authors:
        ego_graph = nx.ego_graph(D, node, undirected=False, distance='inverse_weight')
        ego_graph_density = nx.density(ego_graph)

        closeness_cent = nx.closeness_centrality(G, u = node, distance='inverse_weight')
        
        ego_graph_density_dict.update({node:ego_graph_density})
        closeness_cent_dict.update({node:closeness_cent})
    
    closeness_cent = pd.DataFrame(closeness_cent_dict.items(), columns=['author', 'closeness_cent'])
    ego_graph_density = pd.DataFrame(ego_graph_density_dict.items(), columns=['author', 'ego_graph_density'])
    post_node_merged_df = closeness_cent.merge(ego_graph_density)
    
    # this gives a dataframe for all the nodes' attr. The missing attr for post nodes shoulg have value nan.
    all_node_merged_df = all_node_merged_df.merge(post_node_merged_df, on=['author'], how='left') 
    all_node_merged_df['ego_graph_density'].replace('', np.nan, inplace=True)
    all_node_merged_df.dropna(subset=['ego_graph_density'], inplace=True)
    return all_node_merged_df

"""
Create the adjacency matrix
"""
def get_all_authors(name_id, dataframe):
    """
    A function to get all author names who created the post and replies to the post
    Args:
        name_id (dictionary): id:name pair
        dataframe (): the dataframe to be examined

    Returns:
        author_list: a list of author names
    """

    author_list = set()
    for index, row in dataframe.iterrows():
        uid = str(row['parent_id']).split('_')[-1]
        try:
            author_list.add(name_id[uid])
        except:
            # this would happend if author created the post on a posterior day
            continue

        author_list.update(tuple(row.author))
        
    return author_list

def generate_adj_matrix(name_id, dataframe): 
    """
    A function to generate the adjacency matrix
    Args:
        name_id (dictionary): id:name pair
        dataframe (): the dataframe to be examined

    Returns:
        dataframe: the matrix
    """
    author_list = get_all_authors(name_id, dataframe)

    adj_matrix = pd.DataFrame(0, index = author_list, columns = author_list)

    for index, row in dataframe.iterrows():
        uid = str(row['parent_id']).split('_')[-1]
        try:
            parent_name = name_id[uid]
        except:
            continue
        author_list = row.author

        for author in author_list:
            adj_matrix.loc[author, parent_name] += 1
    
    return adj_matrix

"""
Create the confusion matrix
"""
def confusion_matrix(dataframe):
    """
    A function to create the confusion matrix
    Args:
        dataframe (): the dataframe to e examined

    Returns:
        lists: the lists of the confusion matrix entries
    """
    no_idea_comment = []
    main_post_comment = []
    comment_comment = []
    
    for index, row in dataframe.iterrows():
        if row['main_post'] == np.nan:
            no_idea_comment += row['author']
        elif row['main_post'] == 0:
            comment_comment += row['author']
        else:
            main_post_comment += row['author']
    no_idea_comment = list(dict.fromkeys(no_idea_comment))
    main_post_comment = list(dict.fromkeys(main_post_comment))
    comment_comment = list(dict.fromkeys(comment_comment))
    return no_idea_comment, main_post_comment, comment_comment

"""
Create the type table
"""
def get_type_table(complete_df, no_idea_comment, main_post_comment, comment_comment):
    """
    A function to create the type table
    Args:
        complete_df (): the complete dataframe
        no_idea_comment (list): _description_
        main_post_comment (list): _description_
        comment_comment (list): _description_

    Returns:
        colums of type table and the type table dataframe
    """
    post = complete_df['parent_author'].to_numpy()
    post = set(dict.fromkeys(post))
    post = set(filter(lambda x: x == x , post))
    
    no_idea_comment = set(no_idea_comment)
    main_post_comment = set(main_post_comment)
    comment_comment = set(comment_comment)
    inter_1 = set.intersection(post, no_idea_comment)
    inter_2 = set.intersection(post, main_post_comment)
    inter_3 = set.intersection(post, comment_comment)
    
    post_comment = set().union(inter_1, inter_2, inter_3)
    
    post_only = post - post_comment
    no_idea_comment_only = no_idea_comment - inter_1
    main_post_comment_only = main_post_comment - inter_2
    comment_comment_only = comment_comment - inter_3
    
    classes = ['Members who only post', 'Members who both post and reply to others', 'Members who only reply to main posts', 'Members who only reply to other comments', 'Unidentifiable commenters']
    Counts = [len(post_only), len(post_comment), len(main_post_comment_only), len(comment_comment_only), len(no_idea_comment_only)]
    type_table = pd.DataFrame(list(zip(classes, Counts)), columns = ['Types', 'Counts'])
    
    return type_table, list(post_only), list(post_comment), list(main_post_comment_only), list(comment_comment_only), list(no_idea_comment_only)

"""
Create the type dataframe
"""
def get_type_df(p_o, p_c, m_p_c_o, c_c_o, n_i_c_o):
    """
    A function to create type dataframe
    Args:
        p_o (list): post_only
        p_c (list): post_comment
        m_p_c_o (list): main_post_comment_only
        c_c_o (list): comment_comment_only
        n_i_c_o (list): no_idea_comment_only

    Returns:
        dataframe: the type dataframe
    """
    authors = p_o + p_c + m_p_c_o + c_c_o + n_i_c_o
    post_only = list(np.ones(len(p_o))) + list(np.zeros(len(authors)-len(p_o)))
    post_comment = list(np.zeros(len(p_o))) + list(np.ones(len(p_c))) + list(np.zeros(len(authors)-len(p_o+p_c)))
    reply_main = list(np.zeros(len(p_o+p_c))) + list(np.ones(len(m_p_c_o))) + list(np.zeros(len(c_c_o+n_i_c_o)))
    reply_comment = list(np.zeros(len(authors)-len(c_c_o+n_i_c_o))) +list(np.ones(len(c_c_o))) +list(np.zeros(len(n_i_c_o)))
    unknown = list(np.zeros(len(authors)-len(n_i_c_o))) +list(np.ones(len(n_i_c_o)))
    
    type_df = pd.DataFrame(list(zip(authors, post_only, post_comment, reply_main, reply_comment, unknown)), \
        columns = ['Author Name', 'Members who only post', 'Members who both post and reply to others', 'Members who only reply to main posts', \
        'Members who only reply to other comments', 'Unidentifiable commenters'])
    
    return type_df

def generate_sub_edgelist(month_edgelist, year, month):
    """
    A function that prepares the month edgelist into a list of daily edgelists suitable 
    for computing the daily network parameters
    """
    month_edgelist = month_edgelist[month_edgelist['main_is_old'] != 1]
    month_edgelist = month_edgelist[['Source', 'Target']]
    month_edgelist = dd.from_pandas(month_edgelist, npartitions = 8).groupby(['Source', 'Target']).sum().compute()
    month_edgelist = month_edgelist.groupby(month_edgelist.columns.tolist()).size().reset_index().rename(columns={0:'weight'})
    month_edgelist['inverse_weight'] = 1/month_edgelist['weight']
    month_list = create_subedgelist(year, month, month_edgelist)
    
    return month_list

    
    
def generate_close_centra(month_list, post_name_id, year, month):
    big_df = pd.DataFrame(post_name_id.items(), columns=['id', 'author'])
    big_df = big_df.iloc[: , 1:]
    big_df_set = set(big_df.iloc[:, 0].unique())
    day = 0
    for day_edge in month_list:
        source_target_union = set(day_edge['Source'].unique()).union(set(day_edge['Target'].unique()))
        
        """
        note that this yields the set of authors who have posted a main post in the span of 1.5 year 
        & have posted sth within this day. It doesn't necessarily means that the person has posted a 
        main post on this day!
        """
        post_authors = big_df_set.intersection(source_target_union) 
        
        G = nx.from_pandas_edgelist(edge_list, source = "Source", target = "Target", edge_attr= ["weight", "inverse_weight"], create_using = nx.DiGraph())
        closeness_cent_dict = {}
        for node in post_authors:
            closeness_cent = nx.closeness_centrality(G, u = node, distance='inverse_weight')
            closeness_cent_dict.update({node:closeness_cent})
        closeness_cent = pd.DataFrame(closeness_cent_dict.items(), columns=['author', '{date}'.format(date = str(year) + '_' + str(month) + '_' + str(day))])
        big_df.merge(closeness_cent, how='outer', on='author')
        day += 1
    
    return big_df
        
        
