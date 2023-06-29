
import pandas as pd
import numpy as np
import argparse
import os

def _parse_args():
    
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='combined_plant.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    
    return parser.parse_known_args()


if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    target_col = 'DC_POWER'
    
    # Load data
    df_model_data = pd.read_csv(os.path.join(args.filepath, args.filename), sep=",")
    print(df_model_data.columns)
    print(df_model_data.head(5))

    df_model_data = df_model_data.drop(['SOURCE_KEY', 'DATE_TIME'], axis=1)
    print("after dropped")
    print(df_model_data.columns)
    print(df_model_data.head(5))
    
     # Shuffle and splitting dataset
    train_data, validation_data, test_data = np.split(
        df_model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
    )

    print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
    
    # Save datasets locally
    try:
        train_data.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=False)
        print("Train data saved successfully.")
    except Exception as e:
        print("Error saving train data:", str(e))
    try:
        validation_data.to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=False)
        print("Validation data saved successfully.")
    except Exception as e:
        print("Error saving validation data:", str(e))

    try:
        test_data[target_col].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=False)
        print("Test target data saved successfully.")
    except Exception as e:
        print("Error saving test target data:", str(e))

    try:
        test_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=False)
        print("Test input data saved successfully.")
    except Exception as e:
        print("Error saving test input data:", str(e))

    try:
        df_model_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=False, header=False)
        print("Baseline data saved successfully.")
    except Exception as e:
        print("Error saving baseline data:", str(e))

    
    print("## Processing complete. Exiting.")
