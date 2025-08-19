import argparse
import os
import pandas as pd
import optuna
from utils.model_selection_optuna import get_postgres_db_url
from utils.utils import str2bool


def fetch_study_trials(study_name, save_folder):
    """
    Fetch all trials from an Optuna study and save them as CSV and JSON.
    """

    # Connect to the PostgreSQL database
    storage = get_postgres_db_url()
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Convert trials to DataFrame
    df = study.trials_dataframe()
    
    # Save results as CSV
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, f"{study_name}_trials.csv")
    df.to_csv(csv_path, index=False)
    print(f"Trials saved to: {csv_path}")

    # Extract the best trial parameters
    best_trial = study.best_trial
    best_params = best_trial.params

    # Save best parameters as JSON
    best_params_path = os.path.join(save_folder, f"{study_name}_best_params.json")
    with open(best_params_path, "w") as f:
        pd.DataFrame.from_dict(best_params, orient="index").to_json(f, indent=4)
    print(f"Best parameters saved to: {best_params_path}")


def delete_study(study_name):
    # Define your database URL
    storage_url = get_postgres_db_url()

    # Delete the study
    storage = optuna.storages.RDBStorage(url=storage_url)

    # Get the study ID from the name
    study = optuna.load_study(study_name=study_name, storage=storage)
    study_id = study._study_id  # internal attribute (yes, a bit hacky, but works)

    # Now delete it
    storage.delete_study(study_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and save Optuna study results.")
    parser.add_argument("--study_name", type=str, required=False, help="Name of the Optuna study.")
    parser.add_argument("--save_folder", type=str, required=False, help="Directory to save CSV/JSON results.")
    parser.add_argument("--delete_study", type=str2bool, default=False, help="Delete the study instead fetching results.")
    args = parser.parse_args()

    if args.delete_study:
        delete_study(args.study_name)
        print(f"Study {args.study_name} deleted.")
    else:
        fetch_study_trials(args.study_name, args.save_folder)
        print(f"Study {args.study_name} fetched.")

