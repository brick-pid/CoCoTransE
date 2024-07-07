from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
import os
from typing import Dict

def read_excel(data_path) -> pd.DataFrame:
    """
    Read .xlsx file, and merge all sheet into one(if multiple sheet), return dataframe
    """
    excel_file = pd.ExcelFile(data_path)
    df_list = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(data_path, sheet_name=sheet_name)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    # Check for null values
    if merged_df.isnull().values.any():
        raise ValueError("Null values found in the dataframe.")

    return merged_df

def de_indent(indented: str):
    lines = indented.split('\n')
    if lines:
        # Find the number of leading spaces in the first line
        leading_spaces = len(lines[0]) - len(lines[0].lstrip(' '))
        # Remove the leading spaces from each line
        lines = [line[leading_spaces:] for line in lines]
    return '\n'.join(lines)

def create_cj_dataset(nl: bool = False, patch: bool = False, debug: bool = False) -> Dict[Dataset, Dataset]:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, 'method_parallelData_0322.xlsx')
    patch_01_path = os.path.join(dir_path, 'patch_01.xlsx')
    patch_02_path = os.path.join(dir_path, 'patch_02.xlsx')

    all_data = read_excel(data_path)
    all_data = all_data.applymap(de_indent)

    if debug:
        # get 1% data for debug
        print("-" * 10 + "debugging" "-" * 10)
        all_data = all_data.sample(frac=0.01, random_state=42)

    if nl:
        print("-"*10 + "with NL" + "-" * 10)
        all_data["Java_code"] = all_data[['NL', 'Java_code']].apply(lambda x: '\n'.join(x), axis=1)
    
    # Create a Hugging Face Dataset from the dataframe
    dataset = Dataset.from_pandas(all_data)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if patch:
        print("-"*10 + "with Patch" + "-" * 10)
        patch_01 = read_excel(patch_01_path)
        patch_02 = read_excel(patch_02_path)
        patch = pd.concat([patch_01, patch_02], ignore_index=True)
        patch_dataset = Dataset.from_pandas(patch)
        
        dataset['train'] = concatenate_datasets([dataset['train'], patch_dataset])

    for split in dataset:
        dataset[split] = dataset[split].rename_column("Java_code", "src")
        dataset[split] = dataset[split].rename_column("Cangjie_code", "gt")

    dataset_dict = DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })

    return dataset_dict

def stat_cj():
    # 用于统计仓颉数据集的性质
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, 'method_parallelData_0322.xlsx')
    patch_01_path = os.path.join(dir_path, 'patch_01.xlsx')
    patch_02_path = os.path.join(dir_path, 'patch_02.xlsx')

    all_data = read_excel(data_path)
    all_data = all_data.applymap(de_indent)
    patch_01 = read_excel(patch_01_path)
    patch_02 = read_excel(patch_02_path)
    patch = pd.concat([patch_01, patch_02, all_data], ignore_index=True)

    # Calculate the maximum length of each field
    max_lengths = patch.apply(lambda x: x.str.len()).max()

    # Calculate the minimum length of each field
    min_lengths = patch.apply(lambda x: x.str.len()).min()

    # Calculate the average length of each field
    avg_lengths = patch.apply(lambda x: x.str.len()).mean()

    # Print the results
    print("Maximum Lengths:")
    print(max_lengths)
    print("Minimum Lengths:")
    print(min_lengths)
    print("Average Lengths:")
    print(avg_lengths)




# # # test

if __name__ == "__main__":
    dataset = create_cj_dataset(nl=True, patch=True)
    print(dataset)
    # dataset.save_to_disk('patched_cj_parallel')
    print("-" * 10)
    print(dataset['train'].features)
    print(dataset['train'][0])

    print("-" * 10)
    print('\n', dataset['test'][0])
    
    # print('\n', dataset['train'][-1]['java'])
    # print('\n', dataset['train'][-1]['cangjie'])

    