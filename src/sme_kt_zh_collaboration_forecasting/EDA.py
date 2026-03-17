import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss


# Configure visualization aesthetics for professional reporting
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 12


def perform_general_sales_eda(df: pd.DataFrame):
    """Run a general exploratory analysis of transactional sales data.

    This function performs basic preprocessing, aggregates transactions into a
    regular daily time series, visualizes calendar/seasonal structure, and runs
    simple stationarity diagnostics (ADF/KPSS) to inform downstream modeling.

    Args:
        df: Transaction-level sales data. Must contain a `date` column. If a
            `customer_cat` column is present, it is used for the weekly velocity
            plot by segment.

    Returns:
        daily_series: Daily time series indexed by date with column
            `transaction_count` and added calendar features (month, weekday, etc.).
        mstl_res: Fitted MSTL decomposition result (trend + seasonal components).
        monthly_series: Monthly aggregated transaction counts with columns
            [month, year, transaction_count, month_name].

    Notes:
        - Filters the analysis window to 2019-01-01 <= date < 2026-01-01.
        - Produces multiple plots (Plotly + Matplotlib/Seaborn) and prints test
          statistics to stdout.
    """

    # 1. Data preprocessing and hygiene
    print("--- Phase 1: Data Preprocessing ---")
    data = df.copy()

    # Convert date to datetime and filter to the analysis window.
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    data = data.loc[data["date"] < pd.Timestamp(year=2026, month=1, day=1)].copy()
    data = data.loc[data["date"] > pd.Timestamp(year=2019, month=1, day=1)].copy()

    # Check for and handle duplicates.
    initial_rows = len(data)
    data = data.drop_duplicates()
    print(f"Removed {initial_rows - len(data)} duplicate rows.")

    # 2. Temporal feature extraction
    print("--- Phase 2: Feature Engineering ---")
    data["year"] = data["date"].dt.year
    data["quarter"] = data["date"].dt.quarter
    data["month"] = data["date"].dt.month
    data["month_name"] = data["date"].dt.month_name()
    data["week"] = data["date"].dt.isocalendar().week
    data["day_of_week"] = data["date"].dt.day_name()
    data["day_index"] = data["date"].dt.dayofweek  # 0 = Monday

    # 3. Transactional to regular time-series aggregation
    # Aggregate by day to see the most granular seasonality.
    daily_series = data.groupby("date").size().rename("transaction_count").to_frame()
    # Reindex to ensure no gaps in time.
    full_range = pd.date_range(
        start=daily_series.index.min(),
        end=daily_series.index.max(),
        freq="D",
    )
    daily_series = daily_series.reindex(full_range, fill_value=0)

    # Merge temporal features back to the regularized series.
    daily_series["month"] = daily_series.index.month
    daily_series["month_name"] = daily_series.index.month_name()
    daily_series["day_of_week"] = daily_series.index.day_name()
    daily_series["week"] = daily_series.index.isocalendar().week
    daily_series["quarter"] = daily_series.index.quarter
    daily_series["year"] = daily_series.index.year

    daily_series_wo_weekend = daily_series.loc[
        ~daily_series["day_of_week"].isin(["Saturday", "Sunday"])
    ].copy()

    monthly_series = (
        data.groupby(["month", "year"]).size().rename("transaction_count").to_frame()
    )
    monthly_series.reset_index(inplace=True)
    monthly_series["month_name"] = pd.to_datetime(
        monthly_series["month"].astype(str),
        format="%m",
    ).dt.month_name()

    quarter_series = (
        data.groupby(["quarter", "year"]).size().rename("transaction_count").to_frame()
    )
    quarter_series.reset_index(inplace=True)

    yearly_series = data.groupby(["year"]).size().rename("transaction_count").to_frame()
    yearly_series.reset_index(inplace=True)

    months_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    fig = px.box(
        daily_series,
        x="day_of_week",
        y="transaction_count",
        width=800,
        category_orders={"day_of_week": days_order},
    )
    fig.update_layout(title="Typical Daily Sales Distribution per Day (2018-2026)")
    fig.show()

    fig = px.box(
        daily_series_wo_weekend,
        x="month_name",
        y="transaction_count",
        width=800,
        category_orders={"month_name": months_order},
    )
    fig.update_layout(title="Typical Daily Sales Distribution in a Month (2018-2016)")
    fig.show()

    fig = px.box(daily_series, x="week", y="transaction_count", width=800)
    fig.update_layout(title="Typical Weekly Sales Distribution in a Year (2018-2016)")
    fig.show()

    fig = px.box(
        monthly_series,
        x="month_name",
        y="transaction_count",
        width=800,
        category_orders={"month_name": months_order},
    )
    fig.update_layout(title="Accumulated Sales Distribution in a Month (2018-2026)")
    fig.show()

    fig = px.box(daily_series_wo_weekend, x="quarter", y="transaction_count", width=800)
    fig.update_layout(title="Typical Daily Sales Distribution in a Quarter (2018-2016)")
    fig.show()

    fig = px.box(quarter_series, x="quarter", y="transaction_count", width=800)
    fig.update_layout(title="Accumulated Sales Distribution in a Quarter (2018-2026)")
    fig.show()

    fig = px.box(daily_series_wo_weekend, x="year", y="transaction_count", width=800)
    fig.update_layout(title="Typical Daily Sales Distribution per year")
    fig.show()

    # 4. Heatmap interaction analysis
    pivot_heatmap = daily_series.pivot_table(
        index="month_name",
        columns="day_of_week",
        values="transaction_count",
        aggfunc="mean",
    )

    # Reorder columns and index.
    pivot_heatmap = pivot_heatmap.reindex(index=months_order, columns=days_order)

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_heatmap, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Average Daily Sales Density: Month vs. Weekday")
    plt.show()

    # 5. Statistical decomposition (MSTL)
    print("--- Phase 4: Multi-Seasonal Decomposition ---")
    # Use periods for weekly, monthly-ish, and annual seasonality.
    mstl_model = MSTL(
        daily_series["transaction_count"],
        periods=[7, 12, 365],
        stl_kwargs={"robust": True},
    )
    mstl_res = mstl_model.fit()

    mstl_res.plot()
    plt.suptitle("MSTL Decomposition: Trend + Weekly + Annual + Residuals")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 6. Categorical segmentation EDA
    print("--- Phase 5: Categorical Segment Analysis ---")
    # Use transactional data to inspect customer-type velocity.
    plt.figure(figsize=(14, 6))
    # data['transaction_index'] = data.groupby(['date', 'customer_cat']).cumcount()
    # sns.lineplot(data=data, x='date', y='transaction_index', hue='customer_cat', alpha=0.7)
    # plt.title('Transactional Velocity by Customer Type')
    # plt.ylabel('Transactions per Day (Index)')
    # plt.show()

    all_weeks = (
        data.groupby(["week", "year", "customer_cat"])
        .size()
        .reset_index(name="transaction_count")
    )
    all_weeks["yr_week"] = all_weeks["year"] + all_weeks["week"] / 53.0
    sns.lineplot(data=all_weeks, x="yr_week", y="transaction_count", hue="customer_cat")
    plt.title("Transactional Velocity by Customer Type")
    plt.ylabel("Transactions per Week (Index)")
    plt.show()

    # 7. Stationarity diagnostics
    print("--- Phase 6: Statistical Stationarity Tests ---")
    adf_test = adfuller(daily_series["transaction_count"])
    print(f"ADF Statistic: {adf_test[0]:.4f}")
    print(f"p-value: {adf_test[1]:.4f}")

    if adf_test[1] < 0.05:
        print("Conclusion: Series is stationary at 5% significance.")
    else:
        print("Conclusion: Series is non-stationary; consider differencing.")

    # --- KPSS test (H0: series is stationary) ---
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
        daily_series["transaction_count"],
        regression="c",
    )  # 'c' = level stationarity
    print(f"KPSS statistic: {kpss_stat:.4f}")
    print(f"KPSS p-value:   {kpss_p:.4f}")
    print(f"KPSS lags:      {kpss_lags}")
    print(f"KPSS critical:  {kpss_crit}")
    if kpss_p < 0.05:
        print(
            "Null hypothesis is rejected, meaning that this distribution is non-stationary"
        )
    else:
        print(
            "Null hypothesis is not rejected: The distribution is stationary according to KPSS"
        )

    # ACF / PACF for lag analysis.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    plot_acf(daily_series["transaction_count"], lags=40, ax=ax1)
    plot_pacf(daily_series["transaction_count"], lags=40, ax=ax2)
    plt.show()

    return daily_series, mstl_res, monthly_series


def simple_daily_correlation(sales_df, holiday_df, max_lag):
    """Compute lagged correlation between daily sales counts and holidays.

    The function converts transaction data into a daily sales-count time series
    and creates a binary holiday indicator (1 on holiday dates, 0 otherwise).
    It then computes Pearson correlation for a range of lags by shifting the
    *sales* series relative to the holiday indicator.

    A positive lag corresponds to shifting sales forward in time
    (i.e., comparing holidays to sales that occur *after* the holiday), while a
    negative lag compares holidays to sales *before* the holiday.

    Args:
        sales_df: Transaction-level sales data with a `date` column.
        holiday_df: Holiday calendar with a `date` column. Any date present is
            treated as a holiday (multiple holidays on the same date are
            collapsed into a single indicator).
        max_lag: Maximum lag (in days) to test in both directions.

    Returns:
        best: A single-row Series from the results table corresponding to the
            lag with the largest absolute correlation. Contains keys
            `lag`, `correlation`, and `p_value`.
        results_df: DataFrame with one row per lag and columns
            [`lag`, `correlation`, `p_value`].

    Notes:
        - The p-values are based on the standard t-test for Pearson
          correlation and do not account for autocorrelation, multiple testing
          across lags, or non-stationarity.
        - This is a quick diagnostic, not a causal analysis.
    """

    # Ensure datetime.
    sales_df = sales_df.copy()
    holidays_df = holiday_df.copy()

    # Removing fully closed holidays here would bias the interpretation, so the
    # optional exclusion logic is left commented for experimentation.
    # holidays_df = holidays_df.loc[~holidays_df["holiday name"].isin(["Bundesfeier", "Neujahrstag", "Berchtoldstag", "Weihnachtstag", "Stephanstag"])]

    sales_df["date"] = pd.to_datetime(sales_df["date"])
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    # Aggregate sales to daily counts.
    daily_sales = sales_df.groupby("date").size()

    # Create holiday indicator (1 if holiday, 0 otherwise).
    date_range = pd.date_range(
        min(daily_sales.index.min(), holidays_df["date"].min()),
        max(daily_sales.index.max(), holidays_df["date"].max()),
        freq="D",
    )

    sales_series = daily_sales.reindex(date_range, fill_value=0)
    holiday_series = pd.Series(0, index=date_range)
    holiday_series.loc[holiday_series.index.isin(holidays_df["date"])] = 1

    # Find the optimal lag by testing all lags.
    results = []

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            corr = sales_series.corr(holiday_series)
        else:
            corr = sales_series.shift(lag).corr(holiday_series)

        # Calculate the p-value.
        n = len(sales_series.dropna())
        if not np.isnan(corr) and abs(corr) < 1:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0

        results.append(
            {
                "lag": lag,
                "correlation": corr,
                "p_value": p_value,
            }
        )

    # Find the best lag (highest absolute correlation).
    results_df = pd.DataFrame(results)
    best_idx = results_df["correlation"].abs().idxmax()
    best = results_df.loc[best_idx]
    return best, results_df


# ____________________________________________________________________
# For AutoGluon analysis
#
# Why two `get_train_df*` helpers?
# - `get_train_df(...)` builds the baseline training set (date + target only).
# - `get_train_df_w_holidays(...)` builds the same target series but also adds an
#   `is_holiday` known covariate so we can fit/evaluate AutoGluon models *with*
#   vs *without* holiday information and compare predictive performance.


def get_train_df(df, freq="D"):
    """Aggregate transaction-level sales into a regular time series for modeling.

    Produces a DataFrame compatible with AutoGluon TimeSeries, with one item
    (`item_id = 0`) and a target equal to the number of transactions in each
    resampled period.

    Args:
        df: Transaction-level sales data. Must contain a `date` column that is
            parseable by `pandas.to_datetime`.
        freq: Pandas resampling frequency (e.g., 'D', 'W', 'ME').

    Returns:
        A DataFrame with columns [`date`, `target`, `item_id`], where `date` is a
        regular grid at the specified frequency and missing periods are filled
        with `target = 0`.
    """

    data = df.copy()
    data = data.groupby(["date"]).size().reset_index()
    df = data.set_index("date")

    df = df.resample(freq).agg(
        {
            0: "sum",  # Sum of daily sales for the period.
        }
    )

    full_range = pd.date_range(start=data.date.min(), end=data.date.max(), freq=freq)
    df = (
        df.reindex(full_range, fill_value=0)
        .reset_index()
        .rename(columns={"index": "date", 0: "target"})
    )
    df["item_id"] = 0
    return df


def get_train_df_w_holidays(df, holidays, freq="D"):
    """Aggregate sales to a regular time series and add a holiday covariate.

    Builds a complete daily calendar over the union of sales and holiday dates,
    merges sales counts and a binary holiday indicator, and then resamples to
    the desired frequency.

    Args:
        df: Transaction-level sales data with a `date` column.
        holidays: Holiday calendar with columns `date` and `holiday name`.
            Multiple holidays on the same date are collapsed.
        freq: Pandas resampling frequency (e.g., 'D', 'W', 'ME').

    Returns:
        A DataFrame with columns [`date`, `target`, `is_holiday`, `item_id`].
        `target` is the sum of daily transaction counts over the period.
        `is_holiday` is aggregated with `max` (1 if any holiday occurs in the
        period, else 0).

    Notes:
        - This implementation currently parses dates using format '%d.%m.%Y'.
          If your inputs are already datetime-like or use a different format,
          adjust the `pd.to_datetime(..., format=...)` calls accordingly.
    """

    # --- 1. Normalize dates. ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    holidays = holidays.copy()
    holidays["date"] = pd.to_datetime(holidays["date"], format="%d.%m.%Y")

    # Collapse multiple holidays on the same day.
    holidays = holidays.groupby("date", as_index=False).agg(
        {"holiday name": lambda x: ",".join(sorted(set(x)))}
    )
    holidays["is_holiday"] = 1

    # --- 2. Aggregate raw data to daily sales. ---
    daily_sales = (
        df.groupby("date")
        .size()
        .reset_index(name="target")  # target = number of sales that day
    )

    # --- 3. Build a full daily calendar over the whole span. ---
    start = min(daily_sales["date"].min(), holidays["date"].min())
    end = max(daily_sales["date"].max(), holidays["date"].max())

    calendar = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})

    # --- 4. Merge sales and holidays onto the calendar. ---
    daily = calendar.merge(daily_sales, on="date", how="left").merge(
        holidays[["date", "is_holiday"]], on="date", how="left"
    )

    daily["target"] = daily["target"].fillna(0)  # no sales -> 0
    daily["is_holiday"] = daily["is_holiday"].fillna(0)  # non-holiday days -> 0

    # --- 5. Resample to the desired frequency (D/W/M, etc.). ---
    daily = daily.set_index("date")

    # Holiday aggregation logic:
    # 'max': 1 if any holiday occurs in the period.
    # 'sum': number of holiday days in the period.
    df_res = daily.resample(freq).agg(
        {
            "target": "sum",
            "is_holiday": "max",  # Or 'sum', depending on the use case.
        }
    )

    df_res = df_res.reset_index()
    df_res["item_id"] = 0

    return df_res
