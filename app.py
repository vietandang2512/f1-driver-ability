"""
Streamlit dashboard for the F1 True Driver Ability Measure project.

Reads the same output tables the pipeline scripts in /src produce
(data/processed/master_rankings_2014_present.csv and the two ELO
calibration CSVs) and turns them into something a recruiter can open
in a browser and click around, instead of a spreadsheet they have to
download.

Run locally with: streamlit run app.py
Deployed at: (add your Streamlit Community Cloud URL here once it's live)
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data" / "processed"

st.set_page_config(page_title="F1 True Driver Ability Measure", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,600;0,700;1,700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Titillium Web', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0A0A0A 0%, #15151E 100%);
    }

    /* Racing-stripe banner across the very top of the page */
    .f1-stripe-banner {
        background: repeating-linear-gradient(
            135deg,
            #E10600 0px, #E10600 40px,
            #000000 40px, #000000 80px
        );
        height: 6px;
        width: 100%;
        margin: -1rem 0 1.25rem 0;
        border-radius: 0 0 3px 3px;
    }

    /* Checkered-flag divider, used between major sections */
    .f1-checkered {
        height: 14px;
        width: 100%;
        background-image:
            linear-gradient(45deg, #fff 25%, transparent 25%),
            linear-gradient(-45deg, #fff 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #fff 75%),
            linear-gradient(-45deg, transparent 75%, #fff 75%);
        background-size: 18px 18px;
        background-position: 0 0, 0 9px, 9px -9px, -9px 0px;
        background-color: #000;
        margin: 1.75rem 0;
        border-radius: 2px;
        opacity: 0.9;
    }

    h1 {
        font-weight: 700 !important;
        font-style: italic;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #FFFFFF !important;
        border-left: 6px solid #E10600;
        padding-left: 14px;
    }

    h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E10600;
        padding-bottom: 4px;
        display: inline-block;
    }

    hr {
        border-color: #E10600 !important;
        opacity: 0.35;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #16161F;
        border: 1px solid #2A2A2A;
        border-left: 4px solid #E10600;
        border-radius: 6px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #B8B8B8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    /* Tabs, styled like pit-wall timing screens */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #16161F;
        color: #B8B8B8;
        border-radius: 4px 4px 0 0;
        border: 1px solid #2A2A2A;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E10600 !important;
        color: #FFFFFF !important;
    }

    /* Dataframes and select widgets */
    [data-testid="stDataFrame"] {
        border: 1px solid #2A2A2A;
        border-radius: 6px;
    }

    [data-testid="stCaptionContainer"] {
        color: #9A9A9A !important;
    }
    </style>

    <div class="f1-stripe-banner"></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_master():
    df = pd.read_csv(DATA / "master_rankings_2014_present.csv")
    df["driver_label"] = df["driver"].str.replace("-", " ").str.title()
    rank_cols = ["race_elo_rank", "quali_elo_rank", "race_ridge_rank", "quali_ridge_rank"]
    df["avg_rank"] = df[rank_cols].mean(axis=1, skipna=True)
    return df.sort_values("avg_rank")


@st.cache_data
def load_calibration(filename):
    df = pd.read_csv(DATA / filename)
    df["bucket"] = df["predicted_prob_a"].where(
        df["predicted_prob_a"] >= 0.5, 1 - df["predicted_prob_a"]
    )
    df["favored_won"] = (df["predicted_prob_a"] >= 0.5) == (df["actual_a_won"] == 1.0)
    df["bin"] = (df["bucket"] * 10).astype(int).clip(upper=9)
    return df


def brier_score(df):
    return ((df["predicted_prob_a"] - df["actual_a_won"]) ** 2).mean()


def favorite_accuracy(df):
    decisive = df[df["predicted_prob_a"] != 0.5]
    return (decisive["favored_won"]).mean(), len(decisive)


st.title("F1 True Driver Ability Measure")
st.caption(
    "Who's actually the best driver on the grid, once you control for who has the best car? "
    "Ridge regression and chained ELO, applied to every teammate pairing from 2014 to now."
)

st.markdown('<div class="f1-checkered"></div>', unsafe_allow_html=True)

master = load_master()

st.header("Combined rankings")
st.caption("Sort any column by clicking its header. Lower rank is better, 1 is best of 63.")
display_cols = {
    "driver_label": "Driver",
    "race_elo_rank": "Race ELO rank",
    "quali_elo_rank": "Quali ELO rank",
    "race_ridge_rank": "Race ridge rank",
    "quali_ridge_rank": "Quali ridge rank",
    "avg_rank": "Avg rank",
}
st.dataframe(
    master[list(display_cols.keys())].rename(columns=display_cols).round(1),
    use_container_width=True,
    hide_index=True,
)

st.header("Look up a driver")
driver_choice = st.selectbox("Driver", master["driver_label"])
row = master[master["driver_label"] == driver_choice].iloc[0]

col1, col2, col3, col4 = st.columns(4)
if pd.isna(row["race_elo_rating"]):
    col1.metric("Race ELO", "No data", "not enough pairings")
else:
    col1.metric("Race ELO", f"{row['race_elo_rating']:.0f}", f"rank {int(row['race_elo_rank'])}")

if pd.isna(row["quali_elo_rating"]):
    col2.metric("Qualifying ELO", "No data", "not enough pairings")
else:
    col2.metric("Qualifying ELO", f"{row['quali_elo_rating']:.0f}", f"rank {int(row['quali_elo_rank'])}")

if pd.isna(row["race_ridge_ms"]):
    col3.metric("Race ridge", "No data", "not enough pairings")
else:
    col3.metric("Race ridge", f"{row['race_ridge_ms']:.0f} ms", f"rank {int(row['race_ridge_rank'])}")

if pd.isna(row["quali_ridge_ms"]):
    col4.metric("Qualifying ridge", "No data", "not enough pairings")
else:
    col4.metric("Qualifying ridge", f"{row['quali_ridge_ms']:.0f} ms", f"rank {int(row['quali_ridge_rank'])}")
st.caption(
    "ELO: higher rating is better. Ridge: negative ms means faster than average, "
    "so a more negative number is better."
)

st.markdown('<div class="f1-checkered"></div>', unsafe_allow_html=True)

st.header("Accuracy testing")
st.caption(
    "These numbers come from elo_calibration.py, run against the full 2014-present history. "
    "Every prediction here was made using only races that happened before it."
)

tab_race, tab_quali = st.tabs(["Race pace", "Qualifying pace"])
for tab, filename, label in [
    (tab_race, "elo_calibration_race.csv", "race"),
    (tab_quali, "elo_calibration_quali.csv", "qualifying"),
]:
    with tab:
        cal = load_calibration(filename)
        bs = brier_score(cal)
        acc, n_decisive = favorite_accuracy(cal)

        m1, m2 = st.columns(2)
        m1.metric("Brier score", f"{bs:.3f}", "lower is better, 0.25 = coin flip", delta_color="off")
        m2.metric("Favorite picked correctly", f"{acc:.1%}", f"of {n_decisive} comparisons, baseline 50%")

        bucket_summary = (
            cal.groupby("bin")
            .agg(predicted=("bucket", "mean"), actual=("favored_won", "mean"), n=("bucket", "size"))
            .reset_index()
        )
        bucket_summary["confidence bucket"] = bucket_summary["bin"].apply(lambda b: f"{b*10}-{b*10+10}%")
        st.caption(f"Predicted confidence vs. what actually happened, {label} pace:")
        st.bar_chart(
            bucket_summary.set_index("confidence bucket")[["predicted", "actual"]],
            use_container_width=True,
        )

st.subheader("Chronological holdout (2014-2022 train, 2023 test)")
st.caption(
    "From holdout_test.py, a one-time check, not recomputed on page load since it refits ridge "
    "from scratch. Run the script yourself to reproduce these numbers."
)
holdout_df = pd.DataFrame(
    [
        ["Ridge, race pace", "69.9%", "vs 50% baseline"],
        ["Ridge, qualifying pace", "66.9%", "vs 50% baseline"],
        ["ELO (frozen ratings), race pace", "56.3%", "vs 50% baseline"],
        ["ELO (frozen ratings), qualifying pace", "59.5%", "vs 50% baseline"],
    ],
    columns=["Model", "Picked the faster driver", "Note"],
)
st.dataframe(holdout_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Full methodology, data cleaning rules, limitations, and the problems hit along the way "
    "are in the README. Source: F1DB (github.com/f1db/f1db)."
)
