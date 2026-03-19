from typing import Any

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest


# Shared pipeline helpers


def filter_for_n_orders(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep only customers with at least a minimum number of orders.

    Args:
        df: Transaction-level sales data containing a `customer` column.
        n: Minimum number of orders required for a customer to be retained.

    Returns:
        Filtered copy of `df` containing only customers with at least `n`
        observed orders.
    """
    counts = df.groupby("customer").size()
    keep_customers = counts[counts >= n].index
    return df[df["customer"].isin(keep_customers)].copy()


def create_test_train(
    df: pd.DataFrame, cutoff_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transaction data into train and test sets by date.

    Args:
        df: Transaction-level sales data containing a `date` column.
        cutoff_date: Boundary date used to split the dataset.

    Returns:
        Tuple `(df_train, df_test)` where `df_train` contains rows on or before
        the cutoff date and `df_test` contains rows strictly after it.
    """
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.to_datetime(cutoff_date)
    df_train = df[df["date"] <= cutoff].copy()
    df_test = df[df["date"] > cutoff].copy()
    return df_train, df_test


def prepare_data(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Convert transaction history into a survival-analysis table.

    This function derives inter-purchase durations per customer and augments
    them with a final right-censored observation measuring the time from the
    last purchase to the cutoff date.

    Args:
        df: Transaction-level sales data containing at least `customer` and
            `date` columns.
        cutoff_date: End of the observation window used to define censoring.

    Returns:
        DataFrame with positive durations and an `event` indicator, suitable
        for survival analysis modeling.

    Notes:
        - `event = 1` denotes an observed next order.
        - `event = 0` denotes a censored waiting period ending at `cutoff_date`.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    obs_end_date = pd.to_datetime(cutoff_date)

    # 1. Sort and identify inter-purchase gaps (The Events)
    df = df.sort_values(["customer", "date"])
    df["duration"] = df.groupby("customer")["date"].diff().shift(-1).dt.days
    df["event"] = 1

    # 2. Identify the final 'waiting' period (The Censoring)
    # This captures the time from their last purchase until the cutoff_date
    last_purchases = df.groupby("customer").tail(1).copy()
    last_purchases["duration"] = (obs_end_date - last_purchases["date"]).dt.days
    last_purchases["event"] = 0

    # 3. Combine.
    # Drop rows where duration is NaN (this happens on the last actual purchase row,
    # which is now represented by the censored row we just built).
    final_df = pd.concat(
        [df.dropna(subset=["duration"]), last_purchases], ignore_index=True
    )

    # Clean up: remove 0 or negative durations (same-day orders or orders past cutoff)
    final_df = final_df[final_df["duration"] > 0]

    return final_df


def real_priority_list_from_observed_events(
    df: pd.DataFrame, customer_col: str = "customer"
) -> pd.DataFrame:
    """Build the observed priority ranking from realized next-order times.

    Args:
        df: Prepared survival table containing `duration` and `event` columns.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame with one row per customer, sorted by ascending observed time
        to next order and augmented with `true_time` and `true_rank`.

    Notes:
        Only rows with `event == 1` are used because censored observations do
        not reveal the realized next-order time.
    """
    observed = df[df["event"] == 1].copy()

    per_customer = (
        observed.groupby(customer_col, as_index=False)
        .agg(true_time=("duration", "min"))
        .sort_values("true_time", ascending=True)
        .reset_index(drop=True)
    )
    per_customer["true_rank"] = per_customer.index + 1
    return per_customer


def summarize_top_k_predictions(
    priority_df: pd.DataFrame, top_k: int
) -> dict[str, float]:
    """Summarize overlap between predicted and realized top-k customer ranks.

    Args:
        priority_df: DataFrame containing `pred_rank` ordered by `true_rank`.
        top_k: Number of highest-priority customers to compare.

    Returns:
        Dictionary containing overlap count and recall metrics.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    true_top_k = priority_df.head(top_k).copy()
    correct_predictions = int((true_top_k["pred_rank"] <= top_k).sum())
    overlap_rate = correct_predictions / top_k

    return {
        "top_k": top_k,
        "correct_predictions": correct_predictions,
        "recall_at_k": overlap_rate,
    }


def comparison_row(
    model_name: str, c_index: float, ranking_summary: dict[str, float]
) -> dict[str, float | str]:
    """Build one row for a model-comparison table."""
    return {
        "model": model_name,
        "c_index": c_index,
        "top_k": ranking_summary["top_k"],
        "correct_predictions": ranking_summary["correct_predictions"],
        "recall_at_k": ranking_summary["recall_at_k"],
    }


# Cox helpers


def c_index_on_test_via_score(cph: CoxPHFitter, test_final: pd.DataFrame) -> float:
    """Compute the concordance index using lifelines built-in scoring.

    Args:
        cph: Fitted Cox proportional hazards model.
        test_final: Prepared survival table on which to evaluate the model.

    Returns:
        Concordance index computed by `lifelines` on the provided test data.
    """
    return cph.score(
        test_final.reset_index(drop=True), scoring_method="concordance_index"
    )


def c_index_on_test_manual(cph: CoxPHFitter, test_final: pd.DataFrame) -> float:
    """Compute the concordance index manually from predicted risk scores.

    Args:
        cph: Fitted Cox proportional hazards model.
        test_final: Prepared survival table on which to evaluate the model.

    Returns:
        Concordance index based on predicted partial hazard values.

    Notes:
        The partial hazard is negated so that larger prediction scores align
        with longer survival in `concordance_index`.
    """
    risk = cph.predict_partial_hazard(test_final)
    return concordance_index(
        event_times=test_final["duration"].to_numpy(),
        predicted_scores=(-risk).to_numpy(),
        event_observed=test_final["event"].to_numpy(),
    )


def predicted_priority_list(
    cph: CoxPHFitter, df: pd.DataFrame, customer_col: str = "customer"
) -> pd.DataFrame:
    """Rank customers by predicted urgency to place their next order.

    Args:
        cph: Fitted Cox proportional hazards model.
        df: Prepared survival table containing one or more rows per customer.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame with one row per customer, sorted by descending predicted
        risk and augmented with `pred_risk` and `pred_rank`.

    Notes:
        Only the earliest row per customer is used as the prediction snapshot
        so the ranking does not rely on future covariate information.
    """
    tmp = df.copy()
    tmp = tmp.sort_values([customer_col, "date"])

    first = tmp.groupby(customer_col, as_index=False).head(1).copy()
    first["_risk"] = cph.predict_partial_hazard(first).values

    per_customer = (
        first[[customer_col, "_risk"]]
        .rename(columns={"_risk": "pred_risk"})
        .sort_values("pred_risk", ascending=False)
        .reset_index(drop=True)
    )
    per_customer["pred_rank"] = per_customer.index + 1
    return per_customer


def predicted_vs_real_priorities(
    cph: CoxPHFitter, test_final: pd.DataFrame, customer_col: str = "customer"
) -> pd.DataFrame:
    """Compare predicted customer ranks against realized ordering ranks.

    Args:
        cph: Fitted Cox proportional hazards model.
        test_final: Prepared survival table for the evaluation window.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame joining true and predicted customer rankings for customers
        with observed events in the test set.
    """
    pred = predicted_priority_list(cph, test_final, customer_col=customer_col)
    true = real_priority_list_from_observed_events(
        test_final, customer_col=customer_col
    )
    return true.merge(pred, on=customer_col, how="left").sort_values("true_rank")


# AFT helpers


def predicted_priority_list_aft(
    model: Any, df: pd.DataFrame, customer_col: str = "customer"
) -> pd.DataFrame:
    """Rank customers by predicted urgency using an AFT model.

    Args:
        model: Fitted AFT model exposing `predict_median`.
        df: Prepared survival table containing one or more rows per customer.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame with one row per customer, sorted by ascending predicted
        median time to event and augmented with `pred_time` and `pred_rank`.
    """
    tmp = df.copy()
    tmp = tmp.sort_values([customer_col, "date"])

    first = tmp.groupby(customer_col, as_index=False).head(1).copy()
    first["_pred_time"] = model.predict_median(first).values

    per_customer = (
        first[[customer_col, "_pred_time"]]
        .rename(columns={"_pred_time": "pred_time"})
        .sort_values("pred_time", ascending=True)
        .reset_index(drop=True)
    )
    per_customer["pred_rank"] = per_customer.index + 1
    return per_customer


def predicted_vs_real_priorities_aft(
    model: Any, test_final: pd.DataFrame, customer_col: str = "customer"
) -> pd.DataFrame:
    """Compare AFT-predicted customer ranks against realized ordering ranks.

    Args:
        model: Fitted AFT model exposing `predict_median`.
        test_final: Prepared survival table for the evaluation window.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame joining true and predicted customer rankings for customers
        with observed events in the test set.
    """
    pred = predicted_priority_list_aft(model, test_final, customer_col=customer_col)
    true = real_priority_list_from_observed_events(
        test_final, customer_col=customer_col
    )
    return true.merge(pred, on=customer_col, how="left").sort_values("true_rank")


# RSF helpers


def c_index_rsf(
    rsf: RandomSurvivalForest, X_test: pd.DataFrame, test_final: pd.DataFrame
) -> float:
    """Compute the concordance index for a Random Survival Forest model.

    Args:
        rsf: Fitted Random Survival Forest model.
        X_test: Feature matrix for the test set, aligned to training columns.
        test_final: Prepared survival table containing `duration` and `event`.

    Returns:
        Concordance index based on predicted risk scores.

    Notes:
        RSF predicts higher scores for earlier events, so the sign is negated
        to match the convention expected by `lifelines.concordance_index`.
    """
    risk = rsf.predict(X_test)
    return concordance_index(
        event_times=test_final["duration"].to_numpy(),
        predicted_scores=-risk,
        event_observed=test_final["event"].to_numpy(),
    )


def predicted_priority_list_rsf(
    rsf: RandomSurvivalForest,
    df: pd.DataFrame,
    train_columns: list,
    features: list,
    customer_col: str = "customer",
) -> pd.DataFrame:
    """Rank customers by predicted urgency using a Random Survival Forest.

    Args:
        rsf: Fitted Random Survival Forest model.
        df: Prepared survival table containing one or more rows per customer.
        train_columns: Ordered list of feature columns from training (used to
            align one-hot-encoded test features).
        features: Feature columns to pass to the model; categorical columns
            named `customer_cat` are one-hot encoded.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame with one row per customer, sorted by descending predicted
        risk and augmented with `pred_risk` and `pred_rank`.
    """
    tmp = df.copy()
    tmp = tmp.sort_values([customer_col, "date"])

    first = tmp.groupby(customer_col, as_index=False).head(1).copy()

    X_first = pd.get_dummies(first[features], columns=["customer_cat"], drop_first=True)
    X_first = X_first.reindex(columns=train_columns, fill_value=0)

    first["_risk"] = rsf.predict(X_first)

    per_customer = (
        first[[customer_col, "_risk"]]
        .rename(columns={"_risk": "pred_risk"})
        .sort_values("pred_risk", ascending=False)
        .reset_index(drop=True)
    )
    per_customer["pred_rank"] = per_customer.index + 1
    return per_customer


def predicted_vs_real_priorities_rsf(
    rsf: RandomSurvivalForest,
    test_final: pd.DataFrame,
    train_columns: list,
    features: list,
    customer_col: str = "customer",
) -> pd.DataFrame:
    """Compare RSF-predicted customer ranks against realized ordering ranks.

    Args:
        rsf: Fitted Random Survival Forest model.
        test_final: Prepared survival table for the evaluation window.
        train_columns: Ordered list of feature columns from training.
        features: Feature columns passed to the model.
        customer_col: Column identifying the customer entity.

    Returns:
        DataFrame joining true and predicted customer rankings for customers
        with observed events in the test set.
    """
    pred = predicted_priority_list_rsf(
        rsf, test_final, train_columns, features, customer_col=customer_col
    )
    true = real_priority_list_from_observed_events(
        test_final, customer_col=customer_col
    )
    return true.merge(pred, on=customer_col, how="left").sort_values("true_rank")
