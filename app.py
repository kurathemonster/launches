import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Commercial Space Launches",
    layout="wide",
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/space_devs_launches_cleaned.csv")
    df["launch_datetime"] = pd.to_datetime(df["launch_datetime"], errors="coerce")
    return df


df = load_data()

# Main Settings
st.title("The Commercial Space Boom")
st.caption("Predicting launch growth and success in the modern space industry")

# Setting sidebar
st.sidebar.header("Filters")

year_min = int(df["year"].min())
year_max = int(df["year"].max())
selected_years = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

countries = sorted(df["launch_country"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Launch countries",
    options=countries,
    default=[],
)

provider_types = sorted(df["provider_type"].dropna().unique())
selected_provider_types = st.sidebar.multiselect(
    "Provider types",
    options=provider_types,
    default=[],
)

filtered = df[
    (df["year"] >= selected_years[0])
    & (df["year"] <= selected_years[1])
].copy()

if selected_countries:
    filtered = filtered[filtered["launch_country"].isin(selected_countries)]

if selected_provider_types:
    filtered = filtered[filtered["provider_type"].isin(selected_provider_types)]


# Creating tabs
base_tab, time_tab, geo_tab, success_tab, model_tab = st.tabs(
    [
        "Base Analysis",
        "Time Series",
        "Geography",
        "Success Rates",
        "Prediction Model",
    ]
)

# Base Analysis
with base_tab:
    st.header("Overview")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Launches", f"{len(filtered):,}")
    metric_2.metric("Countries", filtered["launch_country"].nunique())
    metric_3.metric("Providers", filtered["launch_provider"].nunique())
    metric_4.metric("Rockets", filtered["rocket_name"].nunique())

    left_col, right_col = st.columns(2)

    # Launch Status
    with left_col:
        st.subheader("Launch Status")
        status_counts = filtered["status_abbrev"].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        chart = (
            alt.Chart(status_counts).mark_bar()
            .encode(
                x=alt.X('Count:Q'),
                y=alt.Y('Status:N', sort='-x'),
            ).properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)

    # Provider Type
    with right_col:
        st.subheader("Provider Type")
        provider_type_counts = filtered["provider_type"].value_counts().reset_index()
        provider_type_counts.columns = ['Provider Type', 'Count']

        chart = (
            alt.Chart(provider_type_counts).mark_bar()
            .encode(
                x=alt.X('Count:Q'),
                y=alt.Y('Provider Type:N', sort='-x'),
            ).properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)


    display_cols = ['launch_name', 'status', 'launch_provider', 'provider_type', 'rocket_name', 'mission_name',
                    'orbit_abbrev', 'launch_pad', 'launch_location', 'launch_country']

    st.subheader("Filtered Launch Data")
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)



# Time Analysis
with time_tab:
    st.header("Time Series")

    launches_by_year = filtered.groupby("year").size().rename("launches")
    st.subheader("Launches by Year")
    st.line_chart(launches_by_year)

    launches_by_quarter = (
        filtered.groupby(["year", "quarter"])
        .size()
        .reset_index(name="launches")
        .sort_values(["year", "quarter"])
    )
    launches_by_quarter["period"] = (
        launches_by_quarter["year"].astype(str)
        + " Q"
        + launches_by_quarter["quarter"].astype(str)
    )

    st.subheader("Launches by Quarter")
    st.bar_chart(launches_by_quarter, x="period", y="launches")


# Geographical Analysis
with geo_tab:
    st.header("Geography")

    map_data = filtered[["pad_latitude", "pad_longitude"]].dropna().rename(
        columns={"pad_latitude": "lat", "pad_longitude": "lon"}
    )

    st.subheader("Launch Pad Locations")
    if map_data.empty:
        st.info("No mappable launch pad coordinates are available for the selected filters.")
    else:
        st.map(map_data, latitude="lat", longitude="lon", zoom=1)

    st.subheader("Top Launch Countries")
    country_counts = filtered["launch_country"].value_counts().head(15)
    st.bar_chart(country_counts)

    st.subheader("Top Launch Sites")
    site_counts = filtered["launch_location"].value_counts().head(15)
    st.bar_chart(site_counts)


# Success Rates
with success_tab:
    st.header("Success Rates")

    success_df = filtered.dropna(subset=["status_abbrev"]).copy()
    success_df["successful"] = success_df["status_abbrev"].eq("Success")

    success_rate = success_df["successful"].mean()
    st.metric("Overall Success Rate", f"{success_rate:.1%}" if pd.notna(success_rate) else "N/A")

    st.subheader("Success Rate by Provider Type")
    provider_success = (
        success_df.groupby("provider_type")["successful"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(provider_success)

    st.subheader("Success Rate by Launch Country")
    country_success = (
        success_df.groupby("launch_country")["successful"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(15)
    )
    st.bar_chart(country_success["mean"])


# Model Analysis
with model_tab:
    st.header("Prediction Model")
    st.write(
        "Use this tab for model outputs from your notebook: accuracy, confusion matrix, "
        "feature impacts, and example predictions."
    )

    st.subheader("Suggested Model Inputs")
    model_columns = [
        "year",
        "launch_provider",
        "provider_type",
        "rocket_name",
        "mission_type",
        "orbit",
        "launch_country",
    ]
    st.dataframe(filtered[model_columns].head(25), use_container_width=True, hide_index=True)
