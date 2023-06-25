
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

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
    model = linear.Booster()
    model.load_model("linear-model")
    
    # Read test data
    X_test = pd.read_csv(test_x_path, header=None).values
    y_test = pd.read_csv(test_y_path, header=None).to_numpy()

    # Run predictions
    predictions = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Calculate R2 score
    r2 = r2_score(y_test, predictions)
    
    report_dict = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
            },
            "r2_score": {
                "value": r2,
            },
        },
    }

    # Save evaluation report
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
    
    # Save prediction baseline file - we need it later for the model quality monitoring
    pd.DataFrame({"prediction": predictions,
                  "label": y_test.squeeze()}
                ).to_csv(os.path.join(output_prediction_path, 'prediction_baseline/prediction_baseline.csv'), index=False, header=True)
