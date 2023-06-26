
import pandas as pd
import numpy as np
import argparse
import os

def _parse_args():
    
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default, this is an S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='input')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    
    return parser.parse_known_args()


if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    target_col = "DC_POWER"
    
    # Load data
    df_gen1 = pd.read_csv(os.path.join(args.filepath, 'Plant_1_Generation_Data.csv'))
    df_gen2 = pd.read_csv(os.path.join(args.filepath, 'Plant_2_Generation_Data.csv'))
    df_weather1 = pd.read_csv(os.path.join(args.filepath, 'Plant_1_Weather_Sensor_Data.csv'))
    df_weather2 = pd.read_csv(os.path.join(args.filepath, 'Plant_2_Weather_Sensor_Data.csv'))
    
    # Adjust datetime format
    df_gen1['DATE_TIME'] = pd.to_datetime(df_gen1['DATE_TIME'], format='%d-%m-%Y %H:%M')
    df_weather1['DATE_TIME'] = pd.to_datetime(df_weather1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    df_gen2['DATE_TIME'] = pd.to_datetime(df_gen2['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')  # Updated format
    df_weather2['DATE_TIME'] = pd.to_datetime(df_weather2['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # Drop unnecessary columns and merge dataframes
    df_plant1 = pd.merge(
        df_gen1.drop(columns=['PLANT_ID','AC_POWER','TOTAL_YIELD']),
        df_weather1.drop(columns=['PLANT_ID', 'SOURCE_KEY']),
        on='DATE_TIME'
    )

    df_plant2 = pd.merge(
        df_gen2.drop(columns=['PLANT_ID','AC_POWER','TOTAL_YIELD']),
        df_weather2.drop(columns=['PLANT_ID', 'SOURCE_KEY']),
        on='DATE_TIME'
    )

    combined_plant = pd.concat([df_plant1, df_plant2])
    combined_plant.drop(['SOURCE_KEY', 'DATE_TIME'], axis=1)
    
    
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
