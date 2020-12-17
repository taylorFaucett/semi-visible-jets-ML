import pandas as pd
import pathlib

path = pathlib.Path.cwd()


def get_best_sherpa_result(run_type, rinv, objective="Objective"):
    # Use sherpa parameters (same as used in bootstrapping)
    sherpa_file = path / "sherpa_results" / run_type / rinv / "results.csv"
    sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby("Status")
    sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
        by=objective, ascending=False
    )
    best_result = sorted_results.iloc[0].to_dict()
    return best_result


def show_sherpa_results(run_type, rinv, objective="Objective"):
    # Use sherpa parameters (same as used in bootstrapping)
    sherpa_file = path / "sherpa_results" / run_type / rinv / "results.csv"
    sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby("Status")
    sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
        by=objective, ascending=False
    )
    return sorted_results
