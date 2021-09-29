import os
import sys
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple
from cmdstanpy import CmdStanModel

# The participant data directory
data_directory = "individual_participant_data"
# The filenames for the different output files
dataframe_file = "gillan_model.csv"
subject_file = "subjects.csv"
collated_file = "collated.csv"

# The header row of a participants dataframe
column_names = [
    "subj_id",
    "age",
    "gender",
    "trial_num",
    "drift_1",
    "drift_2",
    "drift_3",
    "drift_4",
    "stage_1_response",
    "stage_1_selected",
    "stage_1_RT",
    "transition",
    "stage_2_response",
    "stage_2_selected",
    "stage_2_state",
    "stage_2_RT",
    "reward",
    "redundant"
]

# The columns in the dataframe that should be turned into numbers
number_columns = [
    "trial_num",
    "drift_1",
    "drift_2",
    "drift_3",
    "drift_4",
    "stage_1_selected",
    "stage_1_RT",
    "stage_2_selected",
    "stage_2_state",
    "stage_2_RT",
    "reward",
    "redundant"
]


def response_to_num(resp: str) -> int:
    '''Changes a response (left/right) into a number (0/1)'''
    return int(resp == "right")


def read_file(filename: str) -> Tuple[str, pd.DataFrame]:
    '''
    Reads and processes a participant datafile
    
    This involves skipping the preamble, and reading the rest
    of the data, as well as transforming some columns to numbers.

    Returns a dataframe with header specified above
    '''
    subj_info = [""] * 3
    rows = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # read age, gender, and subject id from the preamble
            if "AgeDropdown" in row:
                subj_info[1] = row[1]
                continue
            if "GenderDropdown" in row:
                subj_info[2] = row[1]
                continue
            if "twostep_instruct_9" in row:
                subj_info[0] = row[0]
                continue
            if len(row) == 15:
                # read trial rows
                rows.append(subj_info + row)
    df = pd.DataFrame.from_records(rows, columns=column_names)
    # Convert columns to numbers
    for col in number_columns:
        df[col] = pd.to_numeric(df[col])
    # Convert responses to numbers
    df['stage_1_response_num'] = df['stage_1_response'].apply(response_to_num)
    df['stage_2_response_num'] = df['stage_2_response'].apply(response_to_num)
    return (subj_info[0], df)


def generate_fit_dataframe(output_dir: str, stan_file: str):
    '''
    Runs the specified model, and writes results to the output directory

    Specifically, this reads all the participant data files into 
    dataframes, collates the necessary inputs to the model, runs
    the model with that input and saves the result
    '''
    subjects = []
    n_trials = []
    first_choices = []
    second_choices = []
    rewards = []
    stage_two_states = []
    # read and collate all of the input files
    for filename in os.listdir(data_directory):
        if Path(filename).suffix != ".csv":
            continue
        full_filename = os.path.join(data_directory, filename)
        subj_id, df = read_file(full_filename)
        subjects.append(subj_id)
        n_trials.append(df.shape[0])
        first_choices.append(df['stage_1_response_num'].values)
        second_choices.append(df['stage_2_response_num'].values)
        rewards.append(df['reward'].values)
        stage_two_states.append(df['stage_2_state'].values)

    # convert each list of ndarrays into a single
    # NumSubj*NumTrial ndarray
    first_choices = np.stack(first_choices)
    second_choices = np.stack(second_choices)
    rewards = np.stack(rewards)
    stage_two_states = np.stack(stage_two_states)

    print(f"subjects = {len(subjects)}, trials = {min(n_trials)}")
    data = {
       "num_subj": len(subjects),
       "num_trials": min(n_trials),
       "s1_responses": first_choices,
       "s2_responses": second_choices,
       "s2_state": stage_two_states,
       "rewards": rewards,
    }
    # compile the model
    gillan_model = CmdStanModel(stan_file=stan_file)
    # perform a penalised MLE of the model on the data
    fit = gillan_model.optimize(data=data)
    # convert the result to a dataframe and save
    df = fit.optimized_params_pd
    dataframe_path = os.path.join(output_dir, dataframe_file)
    df.to_csv(dataframe_path)

    # because the order in which python lists file is undefined,
    # we need to also save a mapping of file number to subject id
    subject_mapping = [[n + 1, id] for n, id in enumerate(subjects)]
    subject_path = os.path.join(output_dir, subject_file)
    with open(subject_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(subject_mapping)


def summarise_datafile(data_dir: str):
    '''
    Reads in the raw data file produced by the model, and outputs
    a csv file with the collated results.
    '''
    filename = os.path.join(data_dir, subject_file)
    # read in the subject mapping
    subjects = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            subjects.append(row)
    # read in the raw dataframe
    filename = os.path.join(data_dir, dataframe_file)
    df = pd.read_csv(filename)
    factors = ["alphas", "beta_mb", "beta_mf0", "beta_mf1", "beta_stick", "beta_s2"]
    rows = []
    # for each subject, build a row with a column for each of the factors
    for [n, subj_id] in subjects:
        row = [subj_id]
        for factor in factors:
            col_name = f"{factor}[{n}]"
            col_mean = df[col_name].values.mean()
            row.append(col_mean)
        rows.append(row)
    header = ["subject"] + factors
    # write a header and the subject rows to a single csv file
    collated_path = os.path.join(data_dir, collated_file)
    with open(collated_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


def main(args: list[str]):
    if len(args) < 2:
        print("missing arguments")
        print("usage: python main.py <output_directory> <model_file>")
        exit(1)

    output_dir = args[0]
    model_file = args[1]
    generate_fit_dataframe(output_dir, model_file)
    summarise_datafile(output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
