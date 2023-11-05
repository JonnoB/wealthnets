import openai
import config  # Import your config.py file
import pandas as pd
import numpy as np
import pickle
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
import time
import re
from datetime import datetime, timedelta
# Set up the OpenAI API key from the config.py file
openai.api_key = config.api_key 

import time
from collections import deque


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

class RateLimiter:
    def __init__(self, max_tokens_per_minute):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens_deque = deque(maxlen=60) # Holds the tokens generated for the past minute.
        self.timestamps_deque = deque(maxlen=60) # Holds the timestamps of when tokens were generated.

    def add_tokens(self, tokens):
        current_time = time.time()

        # Removing tokens older than 1 minute
        while self.timestamps_deque and current_time - self.timestamps_deque[0] > 60:
            self.timestamps_deque.popleft()
            self.tokens_deque.popleft()

        # If the number of tokens is more than the maximum limit,
        # pause execution until it comes back down below the threshold
        if sum(self.tokens_deque) + tokens > self.max_tokens_per_minute:
            sleep_time = 60 - (current_time - self.timestamps_deque[0])
            time.sleep(sleep_time)

            # After sleeping, add the tokens and timestamps to the deque
            self.tokens_deque.append(tokens)
            self.timestamps_deque.append(current_time + sleep_time)
        else:
            # If the number of tokens is less than the maximum limit,
            # add the tokens and timestamps to the deque
            self.tokens_deque.append(tokens)
            self.timestamps_deque.append(current_time)

    def check_tokens(self, tokens):
        # Function to check if adding new tokens would exceed limit, without actually adding them
        current_time = time.time()
        while self.timestamps_deque and current_time - self.timestamps_deque[0] > 60:
            self.timestamps_deque.popleft()
            self.tokens_deque.popleft()

        return sum(self.tokens_deque) + tokens <= self.max_tokens_per_minute
    


def get_model_response(prompt, system_message, rate_limiter, engine="gpt-3.5-turbo"):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    attempts = 0
    while attempts < 5:
        try:
            prompt_length = len(prompt)  # assuming encoding.encode(prompt) is equivalent to len(prompt)
            tokens = len(system_message) + prompt_length
            
            # Add tokens to rate limiter and sleep if necessary
            rate_limiter.add_tokens(tokens)
                
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                max_tokens=2000,
                temperature=0.2,
                top_p=0.9,
            )
            return response['choices'][0]['message']['content'].strip()
            
        except openai.error.RateLimitError as e:
            print(f"RateLimitError encountered: {e}, waiting for a minute...")
            time.sleep(60)  # Wait for a minute before retrying
            continue  # Continue with the next iteration of the loop, thereby retrying the request
            
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}, retrying in 5 seconds...")
            time.sleep(5)

        except openai.error.Timeout as e:
            print(f"TimeoutError encountered: {e}, retrying in 10 seconds...")
            time.sleep(10)
            
        attempts += 1

    print("Failed to get model response after multiple attempts.")
    return None



def identify_gics_classes(df, class_list, system_message, rate_limit=80000, save_path='./data/temp_gics_classes.csv'):
    """
    Identifies GICS classes for each biography in the given DataFrame by making API calls to GPT-3.
    
    The function processes biographies in the DataFrame, identifies relevant GICS classes using GPT-3, 
    and saves the results after every call to a specified path. If the saved file exists, processing
    will resume from the first NA value in the 'classes' column.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing biographies for which GICS classes need to be identified. 
        Expected to have a column named 'Biography'.
        
    class_list : str
        A string containing the list of Global Industry Classification Standard (GICS) industry classes.
        
    system_message : str
        Custom message to handle potential system outputs or errors.
        
    rate_limit : int, optional (default=80000)
        The maximum number of tokens that can be processed per minute as per rate limits.
        
    save_path : str, optional (default='./data/temp_gics_classes.csv')
        The path where the DataFrame is saved after every GPT-3 API call to preserve the progress.

    Returns:
    -------
    temp_df : pd.DataFrame
        A DataFrame identical in structure to `df` but with an additional 'classes' column containing 
        the identified GICS classes for each biography.
        
    Note:
    -----
    Ensure that custom functions like `RateLimiter` and `get_model_response_ethn` are properly defined 
    and accessible in the script where this function is used.

    Examples:
    --------
    >>> df_sample = pd.DataFrame({"Biography": ["John is an expert in finance and works in Wall Street.",
                                                "Jane is a software developer specializing in AI."]})
    >>> classes_list_sample = "Finance, Technology, Healthcare, Real Estate"
    >>> result_df = identify_gics_classes(df_sample, classes_list_sample, "System Error!")
    >>> print(result_df)
    """

    # Check if save_path exists
    if os.path.exists(save_path):
        temp_df = pd.read_csv(save_path)
    else:
        temp_df = df.copy()
        temp_df['classes'] = np.nan

    rate_limiter = RateLimiter(max_tokens_per_minute=rate_limit) 

    n = 100
    total_iterations = temp_df.shape[0]
    #instantiate the Loopmonitor class
    monitor = LoopMonitor(total_iterations, n)
    # Find the first NA value in the 'classes' column
    start_index = temp_df['classes'].isna().idxmax() if temp_df['classes'].isna().any() else 0

    print(f'total_iterations: {total_iterations}, start_index:{start_index}')
    
    for i in range(start_index, total_iterations):

        monitor.update(i)

        biography = temp_df.at[i, 'Biography']

        prompt = f"""
                    Read the list of Global Industry Classification Standard  (GICS) industry classes shown surrounded by 4 colon's below
                    ::::
                    {class_list}
                    ::::

                    Now read the the biography below surrounded by 3 colons, return a python list of GICS classes that most appropriately match the text
                    :::
                    {biography}
                    :::
                    the return string should be in the form shown below and should contain at least 1 entry
                    [entry_1, entry_2,.. entry_n]
                    """

        temp_df.loc[i, 'classes'] = get_model_response(prompt, system_message, rate_limiter, engine="gpt-3.5-turbo")
        
        # Save the process as it goes along. I use csv instead of parwuet as it can be inspected etc, as the program executes
        temp_df.to_csv(save_path, index=False)
    monitor.finish()
    return temp_df

class LoopMonitor:
    """
    A class to monitor and report the progress of a loop, providing
    estimations for completion based on the time taken for the last n iterations.

    Attributes:
    - total_iterations (int): Total number of iterations the loop will run.
    - n (int): The interval at which the class will report the loop's progress.
    - iterations (int): The current number of completed iterations.
    - last_n_start_time (float): The start time of the last n iterations.

    Methods:
    - update(i): To be called inside the loop to update the progress.
    - report(): Prints the current progress and estimations.
    - finish(): To be called after the loop to report the total elapsed time.
    """

    def __init__(self, total_iterations, n):
        """
        Initialize the LoopMonitor with total iterations and report interval.
        """
        self.total_iterations = total_iterations
        self.n = n
        self.iterations = 0

        # Initialize the time for the last n iterations
        # This is crucial because if the loop's tasks vary in execution time,
        # considering only the last n iterations provides a more current 
        # estimation for the remaining time.
        self.last_n_start_time = time.time()

    def update(self, i):
        """
        Update the progress of the loop. This should be called inside the loop.
        """
        self.iterations = i + 1

        if self.iterations % self.n == 0:
            self.report()
            # Reset the time for the next n iterations after reporting
            self.last_n_start_time = time.time()

    def report(self):
        """
        Print a report on the current progress and estimations.
        """
        elapsed_time_for_last_n = time.time() - self.last_n_start_time
        time_per_n = elapsed_time_for_last_n / self.n
        remaining_iterations = self.total_iterations - self.iterations
        remaining_time = time_per_n * remaining_iterations

        # Convert to minutes, round, and then convert back to seconds
        rounded_remaining_time_seconds = round(remaining_time / 60) * 60

        expected_finish_time = (datetime.now() + timedelta(seconds=rounded_remaining_time_seconds)).replace(second=0, microsecond=0)

        print(
            f"Loop number: {self.iterations} | "
            f"Time per iteration: {time_per_n:.4f} seconds | "
            f"Expected finish time: {expected_finish_time}"
        )

    def finish(self):
        """
        Print a report on the total elapsed time. This should be called after the loop.
        """
        total_elapsed_time = time.time() - self.last_n_start_time + (self.iterations / self.n) * (time.time() - self.last_n_start_time)
        print(f"Total elapsed time: {total_elapsed_time:.4f} seconds")




def identify_ethnicity_classes(df, system_message, rate_limit=80000, save_path='./data/temp_ethnicity_classes.csv'):

    # Check if save_path exists
    if os.path.exists(save_path):
        temp_df = pd.read_csv(save_path)
    else:
        temp_df = df.copy()
        temp_df['classes'] = np.nan

    rate_limiter = RateLimiter(max_tokens_per_minute=rate_limit) 

    n = 100
    total_iterations = temp_df.shape[0]
    #instantiate the Loopmonitor class
    monitor = LoopMonitor(total_iterations, n)
    # Find the first NA value in the 'classes' column
    start_index = temp_df['classes'].isna().idxmax() if temp_df['classes'].isna().any() else 0

    for i in range(start_index, total_iterations):

        monitor.update(i)

        row_data = temp_df.loc[i, ['name', 'city']]
        pair_dict = row_data.to_dict()

        prompt = str(pair_dict)

        temp_df.loc[i, 'classes'] = get_model_response(prompt, system_message, rate_limiter, engine="gpt-3.5-turbo")
        
        # Save the process as it goes along. I use csv instead of parwuet as it can be inspected etc, as the program executes
        temp_df.to_csv(save_path, index=False)
    monitor.finish()
    return temp_df