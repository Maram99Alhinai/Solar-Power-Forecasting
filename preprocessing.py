
import pandas as pd
import numpy as np
import argparse
import os

bucket_name = 'sagemaker-us-east-1-531485126105' # my s3

# if have no input this will be default
input_url = f"s3://{bucket_name}/Group-project/Solar1/processing/input/"
output_url = f"s3://{bucket_name}/Group-project/Solar1/processing/output/"

def _parse_args():
    
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default, this is an S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/Group-project/Solar1/processing/input/')
    parser.add_argument('--filename', type=str, default='combined_plant')
    parser.add_argument('--outputpath', type=str, default='Group-project/Solar1/processing/output/')
    
    return parser.parse_known_args()


if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    target_col = "DC_POWER"
    input_rul = input_url
    
    # # Load data
    combined_plant = pd.read_csv(os.path.join(args.filepath, args.filename), sep=";")
    
    # Shuffle and split the dataset
    train_data, validation_data, test_data = np.split(
        combined_plant.sample(frac=1, random_state=1729),
        [int(0.7 * len(combined_plant)), int(0.9 * len(combined_plant))],
    )

    print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
    
    # Save datasets locally
    train_data.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=False)
    validation_data.to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=False)
    test_data[target_col].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=False)
    test_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=False)


    # Save the baseline dataset for model monitoring
    combined_plant.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=False, header=False)
    
    print("## Processing complete. Exiting.")

    
    # # Shuffle the dataset
# shuffled_plant = combined_plant.sample(frac=1).reset_index(drop=True)

# # Split the shuffled dataset
# train_data, validation_data, test_data = np.split(shuffled_plant, [int(0.7 * len(shuffled_plant)), int(0.9 * len(shuffled_plant))])

# print(f"Data split > train: {train_data.shape} | validation: {validation_data.shape} | test: {test_data.shape}")

# print("## Processing complete. Exiting.")

# # Create a folder to save the files
# data_folder = "cleaned_data"  # Specify the data folder path
# os.makedirs(data_folder, exist_ok=True)

# # Define the file paths
# train_x_file = os.path.join(data_folder, "train_x.csv")
# train_y_file = os.path.join(data_folder, "train_y.csv")
# validation_x_file = os.path.join(data_folder, "validation_x.csv")
# validation_y_file = os.path.join(data_folder, "validation_y.csv")
# test_x_file = os.path.join(data_folder, "test_x.csv")
# test_y_file = os.path.join(data_folder, "test_y.csv")
# baseline_file = os.path.join(data_folder, "baseline.csv")
# file_path = os.path.join(data_folder, "combined_plant.csv")

# # Save datasets locally
# train_data.drop([target_col], axis=1).to_csv(train_x_file, index=False)
# train_data[target_col].to_csv(train_y_file, index=False)
# validation_data.drop([target_col], axis=1).to_csv(validation_x_file, index=False)
# validation_data[target_col].to_csv(validation_y_file, index=False)
# test_data.drop([target_col], axis=1).to_csv(test_x_file, index=False)
# test_data[target_col].to_csv(test_y_file, index=False)
# combined_plant.drop([target_col], axis=1).to_csv(baseline_file, index=False)
# combined_plant.to_csv(file_path, index=False)

# print("All files saved successfully to the local folder:", data_folder)
