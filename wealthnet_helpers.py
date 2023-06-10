import pdfplumber
import re
import pandas as pd
import re
from collections import defaultdict
import numpy as np
import copy

def text_to_dataframe(text, keys):
    """
    extracts the summary data from the tables

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
    return s.replace('\n', ' ')

#this function is from https://stackoverflow.com/questions/13781828/replace-a-string-in-list-of-lists
def recursively_apply(l, f):
    for n, i in enumerate(l):
        if type(i) is list:
            l[n] = recursively_apply(l[n], f)
        elif type(i) is str:
            l[n] = f(i)
    return l

#this function deals with the None values that some of the tables produce as headers
def rename_none_in_list(column_names):
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

    extra_columns_indices = [
        i for i, column in enumerate(df_temp.columns) 
        if i != 0 and df_temp.iloc[:, i].dropna().apply(
            lambda x: str(x).split()[-1] if ' ' in str(x) else str(x)
        ).isin({"UHNW", "VHNW", "Deceased", "None"}).all()
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
    elements_pattern = "|".join([fr"{elem}_\d+" for elem in elements_to_clean]) + "|" + "|".join(elements_to_clean)
    
    for element in data_dict.keys():
        if re.match(elements_pattern, element):
            data_dict[element] = clean_unhw_dataframe(data_dict[element])

    return data_dict