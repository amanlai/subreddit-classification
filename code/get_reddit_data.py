import requests
import time
from datetime import datetime
import json
import os
import pandas as pd


def _get_url(is_submission):
    base_url = "https://api.pushshift.io"
    return base_url + f"/reddit/search/{'submission' if is_submission else 'comment'}"



data_list = []


def get_reddit_data(subreddit, batch_size=500, is_submission=True, current_time=None, filename=None):

    """
    Get reddit data as json file using Pushshift API with a more limited number of
    parameters.

    Parameters:
    -------------
    subreddit: str.
        Specify the subreddit to pull data from.

    batch_size: int, default: 500.
        Number of results to return (depending on the other parameters, the number
        of returned results may be fewer). It must be a number divisible by 500.

    is_submission: bool, default: True.
        Return posts if True; comments otherwise.

    current_time: Unix epoch seconds, default: None.
        Return results before this date. If not passed, the current UTC time will 
        be used.

    filename: str, default: None.
        The name of the output json file. If not passed, the subreddit name is 
        used as the file name.

    Output:
    -------------
        None. Saves a json file named as the subreddit name you passed in a folder 
        named "data" in the local environment.

    """

    current_time = int(datetime.now().timestamp()) if current_time is None else current_time

    (rng, size) = (1, batch_size) if batch_size <= 500 else batch_size // 500, 500

    filename = filename if filename is not None else subreddit

    data_list = []

    # api request
    for i in range(rng):

        res = requests.get(
            f"{_get_url(is_submission)}",
            params={
                'subreddit': subreddit,
                'size': size,
                'before': current_time,
            })
        
        # check if the api pull worked
        assert res.status_code == 200, f"Pushshift API data pull did \
        not work as expected. The request returned {res.status_code}."

        # jsonify data and extend the running list with it
        data = res.json().get('data', [])
        data_list.extend(data)

        try:
            current_time = data[-1]['created_utc']
        except IndexError:
            break

        del data

        j = i + 1
        if j < rng:
            time.sleep(3)
        print(f'............ {j / rng * 100:.1f}% done', end='\r')


    with open(f'../data/{filename}.json', 'w') as f:
        json.dump(data_list, f)




def get_dataframe(subreddit, keys=None):
    
    """
    Process json data from a subreddit, get values under specific keys
    and return a pandas dataframe object.
    
    Parameters:
    -------------
    subreddit: str.
        The subreddit whose data to be converted into a dataframe.
        
    keys: list, default: None. 
        The keys to extract from each dictionary in json array. Consult
        Pushshift API for the full list of possible keys. If not passed,
        "subreddit", "created_utc", "selftext" and "title" will be 
        returned.        
    
    Output:
    -------------
    pandas DataFrame.
    """
    
    # the relevant keys to extract
    keys = {'subreddit', 'created_utc', 'selftext', 'title'} if keys is None else keys
    
    # combine json files into a single python list
    lst = []
    for filename in os.listdir('../data/'):
        if filename.endswith('.json') and subreddit in filename:
            with open(f'../data/{filename}', 'r') as f:
                j = json.load(f)
                relevant_kv_pairs = [{k:v for k,v in d.items() if k in keys} for d in j]
                lst.extend(relevant_kv_pairs)
                
    return pd.DataFrame(lst)