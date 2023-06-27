
import json
import os
import pathlib
import pickle as pkl
import tarfile
import joblib
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # All paths are local for the processing container
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_x_path = "/opt/ml/processing/test/test_x.csv"
    test_y_path = "/opt/ml/processing/test/test_y.csv"
    output_dir = "/opt/ml/processing/evaluation"
    output_prediction_path = "/opt/ml/processing/output/"
    
    # Read model tar file
    with tarfile.open(model_path, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = joblib.load("linear-model")
    
    # Read test data
    X_test = pd.read_csv(test_x_path, header=None).values
    y_test = pd.read_csv(test_y_path, header=None).to_numpy().squeeze()

    # Run predictions
    prediction = model.predict(X_test)

    # Get predicted labels (convert probabilities to binary predictions)
    predicted_labels = (prediction > 0.5).astype(int)

    # Evaluate predictions
    rmse_score = np.sqrt(mean_squared_error(y_test, predicted_labels))
    report_dict = {
        "regression_metrics": {
            "rmse_score": {
                "value": rmse_score,
            },
        },
    }

    # Save evaluation report
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))

    # Save prediction baseline file - we need it later for the model quality monitoring
    pd.DataFrame({
        "prediction": predicted_labels,
        "probability": prediction,
        "label": y_test,
    }).to_csv(os.path.join(output_prediction_path, "prediction_baseline/prediction_baseline.csv"), index=False, header=True)
