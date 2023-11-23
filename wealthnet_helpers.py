import pdfplumber
import re
import pandas as pd
import re
from collections import defaultdict
import numpy as np
import copy
import os
import pickle

def text_to_dataframe(text, keys):
    """
    Extract summary data from a text based on specified keys.

    This function takes a `text` string and a list of `keys` as input. It searches for patterns in the text
    to extract summary data associated with each key. The extracted data is returned as a Pandas DataFrame.

    Parameters:
    - text (str): The input text containing summary data.
    - keys (list): A list of keys that represent different sections or categories in the text.

    Returns:
    - pd.DataFrame: A DataFrame with columns corresponding to the provided keys and a single row containing
      the extracted data.

    """
    text = ' '.join([keys[0]] + text.split()) 
    data = {}
    for i, key in enumerate(keys):
        if i < len(keys) - 1:
            pattern = f'{key} (.*?) {keys[i+1]}'
            value = re.search(pattern, text)
            if value:
                data[key] = value.group(1)
        else:
            pattern = f'{key} (.*)'
            value = re.search(pattern, text)
            if value:
                data[key] = value.group(1)
    return pd.DataFrame([data])


def max_depth(lst):
    if not isinstance(lst, list):  # Base case: the input is not a list
        return 0
    elif not lst:  # Base case: empty list
        return 1
    else:
        return 1 + max(max_depth(item) for item in lst)  

#this function is from https://stackoverflow.com/questions/13781828/replace-a-string-in-list-of-lists
def replace_chars(s):
    """ 
    Replace newline characters ('\n') with spaces in a string.

    This function takes a string `s` as input and replaces all newline characters ('\n') in the string with spaces (' ').
    It returns the modified string.

    Parameters:
    - s (str): The input string in which newline characters will be replaced.

    Returns:
    - str: The modified string with newline characters replaced by spaces.
    """
    return s.replace('\n', ' ')

#this function is from https://stackoverflow.com/questions/13781828/replace-a-string-in-list-of-lists
def recursively_apply(l, f):
    """ 
    Recursively apply a function to elements in a nested list.

    This function takes a nested list `l` and a function `f` as input. It iterates through the elements of the list,
    and if an element is a list, it recursively applies the function `f` to each element within the nested list.
    If an element is a string, it applies the function `f` to that string.

    Parameters:
    - l (list): The nested list to apply the function to.
    - f (callable): The function to apply to the elements of the list.

    Returns:
    - list: The nested list with the function `f` applied to its elements.
    """
    for n, i in enumerate(l):
        if type(i) is list:
            l[n] = recursively_apply(l[n], f)
        elif type(i) is str:
            l[n] = f(i)
    return l

#this function deals with the None values that some of the tables produce as headers
def rename_none_in_list(column_names):

    """ 
    Rename 'None' values in a list of column names with unique identifiers.

    This function takes a list of column names `column_names` as input and renames any 'None' values in the list
    with unique identifiers ('None_0', 'None_1', etc.). It ensures that column names are distinct after renaming.

    Parameters:
    - column_names (list): A list of column names, which may contain 'None' values.

    Returns:
    - list: A list of column names with 'None' values replaced by unique identifiers.

    """
    none_count = 0
    new_column_names = []
    for name in column_names:
        if name is None:
            new_column_names.append(f'None_{none_count}')
            none_count += 1
        else:
            new_column_names.append(name)
    return new_column_names

    

def post_process_tables(tables, no_title_tables):
    """
    Post-processes a list of tables by combining table headers with subsequent table bodies, if applicable.

    Args:
        tables (list): A list of tables represented as lists of rows.
        no_title_tables (dict): A dictionary mapping table titles to their corresponding header rows.

    Returns:
        list: A list of post-processed tables where table headers and bodies are combined when necessary.

    """
    processed_tables = []
    prev_table = None
    for table in tables:
        if len(table) == 1:  # This could be a table header split due to a blank line
            for key, value in no_title_tables.items():
                result = set(value).issubset(table[0])
                if result:
                    prev_table = table
                    break
        else:  # This could be a table body or a complete table
            if prev_table:  # If the previous table was a header, combine
                combined_table = prev_table + table
                processed_tables.append(combined_table)
                prev_table = None  # Reset prev_table
            else:
                processed_tables.append(table)  # Append complete table
    return processed_tables

def correct_table_headers(table_temp):
    """
    This function corrects table headers by inserting 'None' where certain columns are identified.

    Parameters:
    -----------
    table_temp : list
        A list of lists, where each sublist represents a row in the table.
        The first sublist is the header row.

    Returns:
    --------
    df_temp : DataFrame
        A pandas DataFrame with corrected headers.

    
    Example:
    --------
    >>> table_temp = [['Name', 'Business', 'Assets'], ['John Doe', 'Tech', 'UHNW'], ['Jane Doe', 'Finance', 'Confirmed VHNW']]
    >>> corrected_df = correct_table_headers(table_temp)
    >>> print(corrected_df)
         Name  Business       None
    0  John Doe      Tech  None UHNW
    1  Jane Doe   Finance  None VHNW

    Note that in this example, 'None' is added to the header row of table_temp to account for 
    the 'None' that appears in the 'Assets' column of the data rows. The 'None' column is thus created.
    """

    # create a temporary dataframe without column headers
    table_temp2 = copy.deepcopy(table_temp)
    df_temp = pd.DataFrame(table_temp2[1:])

    # identify indices of the extra columns
    extra_columns_indices = [
        i for i, column in enumerate(df_temp.columns)
        if i != 0 and df_temp.iloc[:, i].dropna().apply(
            lambda x: bool(re.match("^(Confirmed|Likely) (UHNW|VHNW)|^Deceased$|None", str(x)))
        ).all()
    ]

        # insert 'None' into column names at the identified indices
    for index in extra_columns_indices:
        if table_temp2[0][index] is not None:
            table_temp2[0].insert(index, None)

    # ensure we have the correct number of column names
    assert len(table_temp2[0]) == len(df_temp.columns)

    # assign the corrected headers to the dataframe
    df_temp.columns = table_temp2[0]

    return df_temp


def extract_all_tables(pdf, table2df_params, no_title_tables, verbose = False):
    """
    Extract tables from a PDF document and convert them into a dictionary of Pandas DataFrames.

    This function takes a PDF document `pdf`, table extraction parameters `table2df_params`, and a dictionary
    of tables with no titles `no_title_tables` as input. It extracts tables from the PDF document and converts
    them into Pandas DataFrames. The extracted tables are organized into a dictionary, where each entry is
    identified by a unique table ID.

    Parameters:
    - pdf (PyPDF2.PdfFileReader): A PDF document to extract tables from.
    - table2df_params (pd.DataFrame): A DataFrame containing table extraction parameters, including table names,
      column positions, and data start positions.
    - no_title_tables (dict): A dictionary mapping table names with no titles to their respective column positions.
    - verbose (bool, optional): If True, print verbose output while processing.

    Returns:
    - dict: A dictionary where keys are table IDs and values are Pandas DataFrames containing the extracted tables.
    """

    totalpages = len(pdf.pages)

    tables_dict = {}
    out = None
    table_id = None
    table_counter = {}  # A dictionary to count instances of each table

    for page_number in range(1, totalpages):

        page = pdf.pages[page_number]

        table_settings={
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": page.curves, # used for extract table from page[0]
            "explicit_horizontal_lines": page.curves # used for extract table from page[0]
            }

        table = page.extract_tables(table_settings)
        #removes the \n newline symbold from the text
        table = recursively_apply(table, replace_chars)
        table = post_process_tables(table, no_title_tables)  # Call the post-processing function with no_title_tables
        
        if verbose:
            print("page number "+str(page_number)+" number of tables: " +str(len(table)))

        for n in range(0, len(table)):
            table_temp =table[n]
            if verbose:
                print(table_temp[0][0])
            
            if table_temp and len(table_temp[0]) > 0:
                sub_table = table2df_params.loc[[x == table_temp[0][0] for x in table2df_params['table_name']],:].reset_index(drop=True)
            
                if (sub_table.shape[0]==1):
                    if sub_table['column_names'][0] < len(table_temp):
                        # Add table count handling
                        table_id_base = sub_table['table_id'].to_string(index = False)
                        if table_id_base not in table_counter:
                            table_counter[table_id_base] = 1  # Initialize the counter to 1
                        table_counter[table_id_base] = table_counter.get(table_id_base, 0) + 1
                        table_id = f"{table_id_base}_{table_counter[table_id_base]}"
                        column_names = rename_none_in_list(table_temp[sub_table['column_names'][0]])
                        out = pd.DataFrame(table_temp[sub_table['data_start'][0]:], columns = column_names)
                else:
                    for key, value in no_title_tables.items():
                        result = set(value).issubset(table_temp[0])
                        if result:
                            out = correct_table_headers(table_temp)

                            # Add table count handling
                            table_counter[key] = table_counter.get(key, 0) + 1
                            table_id = f"{key}_{table_counter[key]}"

            if out is not None:
                if verbose:
                    print(table_id)
                    
                tables_dict[table_id] = out
                out = None
                table_id = None
    return tables_dict

def extract_summary_data(pdf):
    """ 
    Extract summary data from a PDF document's second page and convert it into a DataFrame.

    This function takes a PDF document `pdf` as input and extracts structured summary data from its second page.
    The extracted data is converted into a Pandas DataFrame with predefined keys representing different categories.

    Parameters:
    - pdf (PyPDF2.PdfFileReader): A PDF document to extract summary data from.

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted summary data with columns corresponding to predefined keys.

    """

    text_json = pdf.pages[1].extract_text()
    text = text_json.replace("\n", " ")
    keys = ['Name', 'Primary Position', 'Source', 'Primary Company', 'Age', 
            'Estimated Net Worth', 'Birthday', 'Estimated Liquid Assets', 'Marital Status', 'Estimated Household Wealth', 
            'Religion', 'Estimated Household Liquid', 'Alternate Names', "Estimated Family's Net Worth", "Estimated Family's Liquid",
            "Assets", "Wealth Trend", "Residences", "Hometown"]

    df = text_to_dataframe(text, keys)
    return df

def combine_dataframes_with_suffix(dictionary, verbose=False):
    """
    Combines DataFrames in a dictionary based on suffixes.

    Args:
        dictionary (dict): A dictionary containing DataFrames, where the keys represent the suffixes.
        verbose (bool, optional): If True, prints additional information during the process. Default is False.

    Returns:
        dict: A dictionary with combined DataFrames, where the keys represent the prefixes.

    """
    # Group keys with the same prefix
    grouped_keys = defaultdict(list)
    for key in dictionary.keys():
        # Split on the last "_" only if it's followed by a number
        match = re.match(r'^(.*)_([0-9]+)$', key)
        prefix = match.group(1) if match else key
        grouped_keys[prefix].append(key)
    
    if verbose:
        print(grouped_keys)

    combined_dataframes = {}
    for prefix, keys in grouped_keys.items():
        if verbose:
            print(prefix)
        
        # Get non-empty dataframes
        non_empty_dfs = [dictionary[key] for key in keys if not dictionary[key].empty]
        
        if non_empty_dfs:
            combined_df = pd.concat(non_empty_dfs, ignore_index=True)
            combined_dataframes[prefix] = combined_df

    return combined_dataframes


def clean_unhw_dataframe(df):
    """
    Function to clean up the 'Name' column in a DataFrame. It extracts UHNW categories 
    and creates a new column 'uhnw' with these categories. It also removes "deceased" 
    from names and turns all names into lower case. Finally, it removes rows with empty names.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with a 'Name' column to clean.
    
    Returns:
    df (pandas.DataFrame): Cleaned DataFrame.
    """

    df = df.copy()

    # Check if the dataframe has only headers (no data rows)
    if len(df) == 0:
        return df

    #drop columns that are essentially empty
    df= df.replace('', np.nan)
    df = df.dropna(axis=1, how='all')

    # Define UHNW categories
    uhnw_categories = ["confirmed uhnw", "likely uhnw", 
                       "confirmed vhnw", "likely vhnw", "uhnw", "vhnw"]

    # Convert 'Name' to lower case
    df['Name'] = df['Name'].str.lower()

    # Create regex pattern
    pattern = '|'.join(uhnw_categories)

    # Create new column 'uhnw' by extracting category from 'Name'
    df['uhnw'] = df['Name'].apply(lambda x: re.findall(pattern, str(x)) if pd.notnull(x) else None)
    df['uhnw'] = df['uhnw'].apply(lambda x: x[0] if (x and len(x) > 0) else None)

    # Remove UHNW category and "deceased" from 'Name'
    df['Name'] = df['Name'].str.replace(pattern, '', regex=True)
    df['Name'] = df['Name'].str.replace('deceased', '', regex=True)

    # Remove rows with empty 'Name'
    df = df[df['Name'].str.strip() != '']
    
    # rename all columns of name 'None' to a None plus a number as a string. This prevents errors when more than one column is has no Name
    #This missing name error is a side effect of the formatting used when the UNHW indicator was added to the PDF by the organisation.

    counter = 1
    new_columns = []

    for col in df.columns:
        if col is None or re.match('None_\d+', col):
            new_columns.append(f'Unmatched_{counter}')
            counter += 1
        else:
            new_columns.append(col)

    df.columns = new_columns
    
    # Handle 'Unmatched_\d+' columns
    unmatched_cols = [col for col in df.columns if re.match('Unmatched(?:_\d+)?', col)]
    for col in unmatched_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
        df[col + '_uhnw'] = df[col].apply(lambda x: re.findall(pattern, str(x)) if pd.notnull(x) else None)
        df[col + '_uhnw'] = df[col + '_uhnw'].apply(lambda x: x[0] if (x and len(x) > 0) else None)

        # If 'uhnw' in current row is None, but not in 'col + _uhnw', then update 'uhnw' with 'col + _uhnw'
        df['uhnw'] = df.apply(lambda row: row[col + '_uhnw'] if pd.isnull(row['uhnw']) and pd.notnull(row[col + '_uhnw']) else row['uhnw'], axis=1)

    # Drop 'Unmatched_\d+' columns
    df = df.drop(columns=unmatched_cols + [col + '_uhnw' for col in unmatched_cols], errors='ignore')

    df = df.loc[df['Name'].notnull(),:]
    
    return df    


def clean_uhnw_dict_elements(data_dict, elements_to_clean = ['known_associates', 'family_details']):

    """ 
    clean specific elements in a dictionary of data related to Ultra-High-Net-Worth individuals (UHNW).

    This function takes a dictionary `data_dict` and a list of element names `elements_to_clean` as input. It finds the matching dict keys
    and calls `clean_unhw_dataframe`. This is necessary as certain tables are tagged to identify UHNW and the tag causes problems with correct
    table parsing. This function cleans up the problems and makes them machine readable.

    Parameters:
    - data_dict (dict): A dictionary containing data related to UHNW individuals, where keys represent different elements.
    - elements_to_clean (list): A list of element names to be cleaned. By default, it includes 'known_associates' and 'family_details'.
    
    Returns:
    - dict: The input `data_dict` with specified elements cleaned, if found.
    """
    elements_pattern = "|".join([fr"{elem}_\d+" for elem in elements_to_clean]) + "|" + "|".join(elements_to_clean)
    
    for element in data_dict.keys():
        if re.match(elements_pattern, element):
            data_dict[element] = clean_unhw_dataframe(data_dict[element])

    return data_dict
def process_city_files(city_path, new_folder_path, no_title_tables, table2df_params):

    """
    Process PDF files in a city directory and save extracted data as pickled dictionaries.

    This function takes a path to a directory containing PDF files (`city_path`) and a path to a new folder where
    processed data will be saved (`new_folder_path`). It processes each PDF file in the city directory, extracts
    data, and saves it as pickled dictionaries in the new folder. The function also captures processing results
    for each file and returns them as a Pandas DataFrame.

    Parameters:
    - city_path (str): Path to the directory containing PDF files to process.
    - new_folder_path (str): Path to the new folder where processed data will be saved.
    - no_title_tables (dict): A dictionary mapping table names with no titles to their respective column positions.
    - table2df_params (pd.DataFrame): A DataFrame containing table extraction parameters, including table names,

    Returns:
    - pd.DataFrame: A DataFrame containing processing results for each PDF file, including file name, file location,
      and processing status (success or failure).

    Example:
    ```python
    city_directory = '/path/to/city_directory'
    output_folder = '/path/to/output_folder'
    results_df = process_city_files(city_directory, output_folder)
    ```
    In this example, the function processes PDF files in the `city_directory`, saves extracted data in the
    `output_folder`, and returns a DataFrame summarizing processing results for each file.

    Note:
    - This function relies on several other functions like `extract_all_tables`, `extract_summary_data`,
      `clean_uhnw_dict_elements`, and `combine_dataframes_with_suffix` for data extraction and processing.
    - It handles exceptions and captures error messages if processing fails for any file.
    """
    
    target_files = os.listdir(city_path)
    city_name = os.path.basename(city_path)

    city_json_folder = os.path.join(new_folder_path, city_name)
    os.makedirs(city_json_folder, exist_ok=True)

    # Initialize an empty list to store file process results
    results = []

    for file_path in target_files:
        try:
            pdf_file_path = os.path.join(city_path, file_path)
            json_file_name = os.path.splitext(file_path)[0] + '.pkl'
            json_file_path = os.path.join(city_json_folder, json_file_name)
            print(file_path)
            name = re.search(r'Wealth-X (.+?)(?:\sDossier(?:\s\(\d+\))?\.pdf)', file_path).group(1)

            pdf = pdfplumber.open(pdf_file_path)

            tables_dict = extract_all_tables(pdf, table2df_params, no_title_tables)

            tables_dict['summary'] = extract_summary_data(pdf)

            tables_dict = clean_uhnw_dict_elements(tables_dict)

            tables_dict = combine_dataframes_with_suffix(tables_dict)

            with open(json_file_path, 'wb') as file:
                pickle.dump(tables_dict, file)
            
            pdf.close()

            # If no exceptions were raised, then processing was successful
            results.append({
                'file_name': file_path,
                'file_location': pdf_file_path,
                'status': 'success'
            })

        except Exception as e:
            # If an exception was raised, then processing failed
            results.append({
                'file_name': file_path,
                'file_location': pdf_file_path,
                'status': 'failed',
                'error': str(e)  # Capture the error message
            })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    # Return the DataFrame
    return results_df
