import pandas as pd
import streamlit as st
import altair as alt
import joblib
import pydeck as pdk

st.set_page_config(
    page_title="Commercial Space Launches",
    layout="wide",
)

# --------------------------------------------------------------------------------------------------------

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/space_devs_launches_cleaned.csv")
    df["launch_datetime"] = pd.to_datetime(df["launch_datetime"], errors="coerce")
    return df

df = load_data()


# Prophet forecast
@st.cache_data
def load_prophet_forecast():
    forecast = pd.read_csv("artifacts/prophet_forecast.csv")
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    return forecast

@st.cache_data
def load_monthly_actuals():
    actual = pd.read_csv("artifacts/monthly_actual_launches.csv")
    actual['ds'] = pd.to_datetime(actual['ds'])
    return actual


# --------------------------------------------------------------------------------------------------------


# Main Settings
st.title("The Commercial Space Boom")
st.text("Predicting launch growth and success in the modern space industry")

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


# --------------------------------------------------------------------------------------------------------


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

    st.subheader("Launch Data")
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------------------------


# Time Analysis
with time_tab:
    st.header("Time Series")

    # Top Metrics
    scheduled_statuses = ["Go", "TBC", "TBD"]
    completed_statuses = ["Success", "Failure", "Partial Failure"]

    scheduled_launches = filtered["status_abbrev"].isin(scheduled_statuses).sum()
    completed_launches = filtered["status_abbrev"].isin(completed_statuses).sum()

    scheduled_col, completed_col = st.columns(2)
    scheduled_col.metric("Scheduled Launches", f"{scheduled_launches:,}")
    completed_col.metric("Completed Launches", f"{completed_launches:,}")


    # Launches Over Time
    st.subheader("Launches by Year")
    
    launches_by_year = filtered.groupby("year").size().reset_index(name="launches")
    launches_by_year['year'] = launches_by_year['year'].astype(str)

    year_chart = (alt.Chart(launches_by_year)
                  .mark_line(point=True)
                  .encode(
                      x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                      y=alt.Y('launches:Q', title='Launches'),
                  ).properties(height=400)
                )

    st.altair_chart(year_chart, use_container_width=True)

    st.text('There is a steady increase in launches over the years. No particular seasonality seen in the data.')
    st.text('Note: This graph counts the 163 planned launches remaining in 2026.')


    # Predicted Launches (prophet graph)
    actual = load_monthly_actuals()
    forecast = load_prophet_forecast()
    
    actual_points = (
        alt.Chart(actual)
        .mark_circle(size=45)
        .encode(
            x=alt.X('ds:T', title='Year'),
            y=alt.Y('y:Q', title='Launches'),
            color=alt.value("#ff376f"),
            tooltip=[
                alt.Tooltip('ds:T', title='Month'),
                alt.Tooltip('y:Q', title='Actual Launches')
            ],
        )
    )

    forecast_band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X('ds:T', title='Year'),
            y=alt.Y('yhat_lower:Q', title='Launches'),
            y2='yhat_upper:Q',
        )
    )

    forecast_line = (
        alt.Chart(forecast)
        .mark_line()
        .encode(
            x=alt.X("ds:T", title="Year"),
            y=alt.Y('yhat:Q', title='Launches'),
            tooltip=[
                alt.Tooltip('ds:T', title='Date'),
                alt.Tooltip('yhat:Q', title='Prediction', format='.1f'),
                alt.Tooltip('yhat_lower:Q', title='Lower Bound', format='.1f'),
                alt.Tooltip('yhat_upper:Q', title='Upper Bound', format='.1f')
            ],
        )
    )
    
    st.subheader("Predicted Launches")
    
    chart = (forecast_band + forecast_line + actual_points).properties(height=500)
    st.altair_chart(chart, use_container_width=True)

    st.text('THe number of launches are expected to increase in the coming years.')


# --------------------------------------------------------------------------------------------------------


# Geographical Analysis 

with geo_tab:
    st.header("Geography")
    st.text('This section takes a look at the top sites and area of the world where launches occur.')


    # First Geographical Map - Launch Pad Locations
    st.subheader("Launch Pad Locations")
    
    map_data = filtered[["pad_latitude", "pad_longitude"]].dropna().rename(
        columns={"pad_latitude": "lat", "pad_longitude": "lon"}
    )

    if map_data.empty:
        st.info("No mappable launch pad coordinates are available for the selected filters.")
    else:
        st.map(map_data, latitude="lat", longitude="lon", zoom=1)



    # Second geographical map - Launch Volume by Location
    st.subheader('Launch Volume by Site')

    site_volume = (filtered.dropna(subset=['pad_latitude', 'pad_longitude']).groupby(['launch_location', 'launch_country'])
                   .agg(
                       launches=('launch_id', 'count'),
                       pad_latitude=('pad_latitude', 'mean'),
                       pad_longitude=('pad_longitude', 'mean'),
                   ).reset_index()
    )
    if site_volume.empty:
        st.info('No launch coordinates available for selected filters.')
    else:
        max_launches = site_volume['launches'].max()

        def launch_color(count):
            ratio = count / max_launches

            if ratio > 0.75:
                return [190, 30, 45, 190]      # red: highest activity
            elif ratio > 0.40:
                return [230, 125, 35, 180]     # orange
            elif ratio > 0.15:
                return [240, 200, 80, 170]     # yellow
            else:
                return [120, 170, 150, 150]    # muted green

    site_volume['color'] = site_volume['launches'].apply(launch_color)

    site_layer = pdk.Layer("ColumnLayer", data=site_volume, get_position=['pad_longitude', 'pad_latitude'], get_elevation='launches',
                           elevation_scale=5000, radius=250000, get_fill_color='color', pickable=True, auto_highlight=True)

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=25, bearing=0)

    tooltip = {
        "html": """
    <b>{launch_location}</b><br/>
    Country: {launch_country}<br/>
    Launches: {launches}
    """,
    "style": {
        "backgroundColor": "white",
        "color": "black",
        },
    }

    deck = pdk.Deck(layers=[site_layer], initial_view_state=view_state,
                        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                        tooltip=tooltip)

    st.pydeck_chart(deck, use_container_width=True, height=600)




    # Bar Graphs for Top Sites
    st.subheader("Top Launch Countries")
    country_counts = filtered["launch_country"].value_counts().head(15).reset_index()
    country_counts.columns = ['country', 'launches']

    country_chart = (alt.Chart(country_counts).mark_bar()
                     .encode(
                         x=alt.X("launches:Q", title='Launches'),
                         y=alt.Y('country:N', title='Country', sort='-x'),
                         tooltip=[
                             alt.Tooltip('country:N', title='Country'),
                             alt.Tooltip('launches:Q', title='Launches'),
                         ],
                     ).properties(height=450)
    )

    st.altair_chart(country_chart, use_container_width=True)


    st.subheader("Top Launch Sites")
    site_counts = filtered['launch_location'].value_counts().head(15).reset_index()
    site_counts.columns = ['site', 'launches']

    site_chart = (alt.Chart(site_counts).mark_bar()
                  .encode(
                      x=alt.X('launches:Q', title='Launches'),
                      y=alt.Y('site:N', title='Launch Site', sort='-x'),
                      tooltip=[
                          alt.Tooltip('site:N', title='Launch Site'),
                          alt.Tooltip('launches:Q', title='Launches'),
                      ],
                  ).properties(height=500)
    )

    st.altair_chart(site_chart, use_container_width=True)


# --------------------------------------------------------------------------------------------------------


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

# --------------------------------------------------------------------------------------------------------

# Model Analysis
@st.cache_resource
def load_success_model():
    model = joblib.load("../artifacts/success_model_pipeline.joblib")
    features = joblib.load("../success_model_features.joblib")
    return model, features



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
