import pandas as pd
import streamlit as st
import altair as alt
import joblib
import plotly.express as px
import pydeck as pdk
import ast
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

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
st.text("Predicting Launch Growth and Success in the Modern Space Industry")
st.caption('How have geography, launch provider, rocket type, and mission characteristics shaped global launch growth since 2010, and how well can those patterns predict future launch volume and launch success?')


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
base_tab, time_tab, geo_tab, model_tab, summary_tab = st.tabs(
    [
        "Base Analysis",
        "Time Series",
        "Geography",
        "Prediction Model",
        "Summary",
    ]
)


# --------------------------------------------------------------------------------------------------------


# Base Analysis
with base_tab:
    st.write("*Space is the way of the future.*")
    st.write(
        "Commercial space launches are no longer rare government milestones. "
        "Since 2010, they have become a fast-growing global industry shaped by "
        "geography, private providers, rocket systems, and mission goals. "
        "Understanding these patterns matters because they help show not only "
        "where the space economy is growing, but also what factors may influence "
        "future launch success."
    )

    img = Image.open("img/spaceship.jpg")
    st.image(img)

    st.divider()


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

# Add provider experience vs. launch success
# launch volume over time by provider type
# Add top launch providers



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

    st.text('There is a steady increase in launches over the years. No particular pattern for seasonality seen in the data.')
    st.write('*Note: This graph counts the 163 planned launches remaining in 2026.*')


    st.divider()

    # Predicted Launches (prophet graph)
    actual = load_monthly_actuals()
    forecast = load_prophet_forecast()
    
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['trend_lower'] = forecast['trend_lower'].clip(lower=0)
    forecast['trend'] = forecast['trend'].clip(lower=0)
    year_axis = alt.X(
        "ds:T",
        title="Year",
        axis=alt.Axis(format="%Y", tickCount="year"),
    )

    actual_points = (
        alt.Chart(actual)
        .mark_circle(size=45)
        .encode(
            x=year_axis,
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
            x=year_axis,
            y=alt.Y('trend_lower:Q', title='Launches'),
            y2='trend_upper:Q',
        )
    )

    forecast_line = (
        alt.Chart(forecast)
        .mark_line()
        .encode(
            x=year_axis,
            y=alt.Y('trend:Q', title='Launches'),
            tooltip=[
                alt.Tooltip('ds:T', title='Date'),
                alt.Tooltip('trend:Q', title='Predicted Trend', format='.1f'),
                alt.Tooltip('trend_lower:Q', title='Lower Bound', format='.1f'),
                alt.Tooltip('trend_upper:Q', title='Upper Bound', format='.1f')
            ],
        )
    )
    
    st.subheader("Predicted Launch Trend")
    
    chart = (forecast_band + forecast_line + actual_points).properties(height=500)
    st.altair_chart(chart, use_container_width=True)

    st.write(
        "This plot shows launch counts against Prophet's long-term trend "
        "forecast. The values vary, but the main pattern is that "
        "launch volume is expected to increase in the coming years."
    )


# --------------------------------------------------------------------------------------------------------


# Geographical Analysis 

with geo_tab:
    st.header("Geography")
    st.text('This section takes a look at the top sites and areas of the world where launches occur.')


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

with model_tab:
    st.header("Success Rates")

    # Completed Launch Success Rate
    success_df = filtered[
        filtered["status_abbrev"].isin(["Success", "Failure", "Partial Failure"])
    ].copy()
    success_df["successful"] = success_df["status_abbrev"].eq("Success")

    completed_success_rate = success_df["successful"].mean()
    st.metric(
        "Completed Launches Success Rate",
        f"{completed_success_rate:.1%}" if pd.notna(completed_success_rate) else "N/A",
    )


    # Launch Success Counts by Country
    st.subheader("Launch Success Counts by Country")
    st.caption("First we'll look at the success rates for launches of each country.")
    country_launches = (success_df.groupby("launch_country")
        .agg(
            total_launches=("launch_id", "count"),
            successful_launches=("successful", "sum"),
        ).query("total_launches >= 10")
        .assign(success_rate=lambda data: (data["successful_launches"] / data["total_launches"] * 100).round(2))
        .sort_values("total_launches", ascending=False).reset_index()
    )

    if country_launches.empty:
        st.info("No countries have at least 10 launches for the selected filters.")
    else:
        country_launches_long = country_launches.melt(
            id_vars=["launch_country", "success_rate"],
            value_vars=["total_launches", "successful_launches"],
            var_name="launch_type",
            value_name="launches",
        )

        country_launch_chart = px.bar(
            country_launches_long,
            x="launch_country",
            y="launches",
            color="launch_type",
            barmode="group",
            title="Total Launches",
            labels={
                "launch_country": "Country",
                "launches": "Number of Launches",
                "launch_type": "Launch Type",
                "success_rate": "Success Rate",
            },
            category_orders={
                "launch_country": country_launches["launch_country"].tolist(),
                "launch_type": ["total_launches", "successful_launches"],
            },
            color_discrete_map={
                "total_launches": "#4c78a8",
                "successful_launches": "#f58518",
            },
            hover_data={"success_rate": ":.2f"},
        )
        country_launch_chart.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(country_launch_chart, use_container_width=True)
        st.dataframe(country_launches, use_container_width=True, hide_index=True)

    st.text('The US has the most amount of launches, but their success rate is not the highest. The country with the highest success rate is Russia.')


    st.divider()

# --------------------------------------------------------------------------------------------------------

    # Model Analysis

    @st.cache_resource
    def load_success_model():
        model = joblib.load("artifacts/success_model_pipeline.joblib")
        features = joblib.load("artifacts/success_model_features.joblib")
        return model, features

    @st.cache_data
    def get_feature_impacts():
        model, _ = load_success_model()
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        coefficients = model.named_steps["classifier"].coef_[0]

        feature_impacts = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
            }
        )
        feature_impacts["abs_coefficient"] = feature_impacts["coefficient"].abs()
        feature_impacts["direction"] = feature_impacts["coefficient"].apply(
            lambda value: "Predicts Success" if value > 0 else "Predicts Failure"
        )

        readable_features = (
            feature_impacts["feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
            .str.replace("launch_provider_", "Provider: ", regex=False)
            .str.replace("provider_type_", "Provider Type: ", regex=False)
            .str.replace("rocket_family_", "Rocket Family: ", regex=False)
            .str.replace("rocket_full_name_", "Rocket: ", regex=False)
            .str.replace("mission_type_", "Mission Type: ", regex=False)
            .str.replace("orbit_abbrev_", "Orbit: ", regex=False)
            .str.replace("launch_country_", "Country: ", regex=False)
            .str.replace("launch_location_", "Location: ", regex=False)
            .str.replace("launch_pad_", "Pad: ", regex=False)
            .str.replace("date_precision_", "Date Precision: ", regex=False)
            .str.replace("_", " ", regex=False)
            .str.replace("yearly provider attempt number", "Yearly Provider Number of Attempts", regex=False)
        )
        feature_impacts["feature_label"] = readable_features
        return feature_impacts.sort_values("abs_coefficient", ascending=False)


    st.header("Prediction Model")
    st.write(
        "This logistic regression model estimates whether a launch is likely to succeed based on launch provider, rocket, orbit, location, and launch history features."
    )


# --------------------------------------------------------------------------------------------------------

    # List of Feature Importances

    st.subheader("What Predicts Success?")
    st.caption(
        "Positive coefficients push the model toward predicting success. Negative coefficients push it toward predicting failure. These are associations in the training data, not proof of direct causes."
    )

    feature_impacts = get_feature_impacts()

    impact_view = st.radio(
        "Feature impact view",
        ["Strongest overall", "Success predictors", "Failure predictors"],
        horizontal=True,
    )

    if impact_view == "Success predictors":
        chart_data = feature_impacts[feature_impacts["coefficient"] > 0].head(20)
    elif impact_view == "Failure predictors":
        chart_data = feature_impacts[feature_impacts["coefficient"] < 0].head(20)
    else:
        chart_data = feature_impacts.head(20)

    chart_data = chart_data.sort_values("coefficient")

    impact_chart = px.bar(
        chart_data,
        x="coefficient",
        y="feature_label",
        color="direction",
        orientation="h",
        title="Top Logistic Regression Feature Impacts",
        labels={
            "coefficient": "Model Coefficient",
            "feature_label": "Feature",
            "direction": "Direction",
            "abs_coefficient": "Impact Size",
        },
        color_discrete_map={
            "Predicts Success": "#2a9d8f",
            "Predicts Failure": "#e76f51",
        },
        hover_data={
            "feature": True,
            "abs_coefficient": ":.3f",
            "feature_label": False,
            "direction": True,
        },
    )
    impact_chart.update_layout(
        height=650,
        yaxis_title=None,
        legend_title_text="Model Direction",
    )
    st.plotly_chart(impact_chart, use_container_width=True)

    st.write("""Test flights and GSTO have the highest negative impact on flight success.
            Geostationary Transfer Orbit (GTO) is typically used as an intermediate orbit for satellites destined for GEO. These missions are typically more demanding in power and precision, which could result in a higher failure rate.""")
    
    st.write("""Positive predictors for successful launches include launch provider ULA, launches destined for MEO, and providers with more launch attempts in that year.
             Specific rockets being named for having a negative impact should be taken with caution, as those values likely are rare and are impacting the data greatly.""")
    
# --------------------------------------------------------------------------------------------------------
    
    # Model Metrics
    
    st.divider()

    st.subheader("Predicting Success")

    @st.cache_data
    def load_success_model_data():
        return pd.read_csv("data/success_model_cleaned.csv")
    
    model_df = load_success_model_data()

    model, features = load_success_model()
    
    feature_cols = [
        "launch_provider",
        "provider_type",
        "rocket_family",
        "rocket_full_name",
        "mission_type",
        "orbit_abbrev",
        "launch_country",
        "launch_location",
        "launch_pad",
        "date_precision",
        "year",
        "provider_attempt_number",
        "pad_attempt_number",
        "location_attempt_number",
        "yearly_provider_attempt_number",
        "yearly_pad_attempt_number",
        "yearly_location_attempt_number",
        "pad_total_launch_count",
        "pad_orbital_attempt_count",
        "location_total_launch_count",
]

    X = model_df[feature_cols]
    y = model_df['success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    failure_recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Precision", f"{precision:.1%}")
    col3.metric("Success Recall", f"{recall:.1%}")
    col4.metric("F1 Score", f"{f1:.1%}")
    col5.metric("Failure Recall", f"{failure_recall:.1%}")
    

    matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])

    confusion = pd.DataFrame(
        matrix,
        index=["Actual Success", "Actual Failure"],
        columns=["Predicted Success", "Predicted Failure"],
    )

    st.dataframe(confusion)

    st.write(
        "The largest problem encountered with this dataset is that the target to "
        "predict launch success is too small to create an accurate prediction. "
        "While the model can confidently say that a launch will succeed, the model "
        "fails to predict the actual failures because there is too little data to "
        "work off of, as shown by the low failure recall value."
    )

# --------------------------------------------------------------------------------------------------------

    # Model tester

    st.divider()

    st.subheader("Test the Success Model")

    model, features = load_success_model()
    model_df = load_success_model_data()


    # Cleaning rocket_family
    def parse_rocket_family(value):
        if pd.isna(value) or value == "[]":
            return {"name": "Unknown", "id": "Unknown"}
        
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list) and len(parsed) > 0:
                item = parsed[0]
                return {"name": item.get("name", "Unknown"), "id": item.get("id", "Unknown")}
            
            if isinstance(parsed, dict):
                return {"name": parsed.get("name", "Unknown"), "id": parsed.get("id", "Unknown")}
            
        except (ValueError, SyntaxError):
            pass

        return {"name": str(value), "id": "Unknown"}
    
    def format_rocket_family(value):
        parsed = parse_rocket_family(value)

        if parsed["id"] == "Unknown":
            return parsed['name']
        
        return f"{parsed['name']} (ID: {parsed['id']})"


    # Creatring dropdown options
    launch_provider = st.selectbox(
        "Launch Provider",
        sorted(model_df['launch_provider'].dropna().unique())
    )

    mission_type = st.selectbox(
        "Mission Type",
        sorted(model_df['mission_type'].dropna().unique())
    )

    orbit_abbrev = st.selectbox(
        "Orbit",
        sorted(model_df['orbit_abbrev'].dropna().unique())
    )

    launch_country = st.selectbox(
        "Launch Country",
        sorted(model_df['launch_country'].dropna().unique())
    )

    rocket_family_options = sorted(model_df['rocket_family'].dropna().unique(), key=format_rocket_family)

    rocket_family = st.selectbox(
        "Rocket Family", rocket_family_options, format_func=format_rocket_family
    )

    yearly_provider_attempt_number = st.slider(
        "Provider Launch Attempts This Year",
        min_value=0,
        max_value=int(model_df['yearly_provider_attempt_number'].max()),
        value=int(model_df['yearly_provider_attempt_number'].median())
    )

    # Defining columns
    categorical_features = [
        "launch_provider",
        "provider_type",
        "rocket_family",
        "rocket_full_name",
        "mission_type",
        "orbit_abbrev",
        "launch_country",
        "launch_location",
        "launch_pad",
        "date_precision",
    ]

    numeric_features = [
        "year",
        "provider_attempt_number",
        "pad_attempt_number",
        "location_attempt_number",
        "yearly_provider_attempt_number",
        "yearly_pad_attempt_number",
        "yearly_location_attempt_number",
        "pad_total_launch_count",
        "pad_orbital_attempt_count",
        "location_total_launch_count",
    ]

    # Creating default values
    default_input = {}

    for col in categorical_features:
        default_input[col] = model_df[col].fillna("Unknown").mode()[0]

    for col in numeric_features:
        numeric_col = pd.to_numeric(model_df[col], errors="coerce")
        default_input[col] = numeric_col.median()
    
    # Overwrite use controlled features
    default_input['launch_provider'] = launch_provider
    default_input["mission_type"] = mission_type
    default_input["orbit_abbrev"] = orbit_abbrev
    default_input["rocket_family"] = rocket_family
    default_input["yearly_provider_attempt_number"] = yearly_provider_attempt_number

    user_input = pd.DataFrame([default_input])

    # Predict
    if st.button("Predict Success"):
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]

        st.metric("Predicted Success Probability", f"{probability:.1%}")

        if prediction == 1:
            st.success("The model predicts this launch is likely to succeed.")
        else:
            st.warning("The model predicts this launch may have higher failure risk.")


# --------------------------------------------------------------------------------------------------------

# Summary

with summary_tab:
    st.write(
        "As space launches become more frequent and commercially driven, "
        "understanding the patterns behind them becomes increasingly important. "
        "The data shows that while launch success depends on complex and "
        "sometimes rare factors, the overall direction of the industry is "
        "unmistakable: more launches, more competition, continued growth."
    )

    st.subheader("Summary Snapshot")

    completed_filtered = filtered[
        filtered["status_abbrev"].isin(["Success", "Failure", "Partial Failure"])
    ].copy()
    completed_success_rate = completed_filtered["status_abbrev"].eq("Success").mean()
    top_country = filtered["launch_country"].mode()
    top_provider = filtered["launch_provider"].mode()
    summary_launches_by_year = filtered.groupby("year").size().reset_index(name="launches")

    if (
        len(summary_launches_by_year) > 1
        and summary_launches_by_year["launches"].iloc[0] > 0
    ):
        growth_rate = (
            (
                summary_launches_by_year["launches"].iloc[-1]
                - summary_launches_by_year["launches"].iloc[0]
            )
            / summary_launches_by_year["launches"].iloc[0]
        )
        growth_label = f"{growth_rate:.0%}"
    else:
        growth_label = "N/A"

    summary_1, summary_2, summary_3, summary_4 = st.columns(4)
    summary_1.metric("Launch Growth", growth_label)
    summary_2.metric(
        "Completed Success Rate",
        f"{completed_success_rate:.1%}" if pd.notna(completed_success_rate) else "N/A",
    )
    summary_3.metric("Top Country", top_country.iloc[0] if not top_country.empty else "N/A")
    summary_4.metric("Top Provider", top_provider.iloc[0] if not top_provider.empty else "N/A")

    snapshot_chart = (
        alt.Chart(summary_launches_by_year)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("launches:Q", title="Launches"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("launches:Q", title="Launches"),
            ],
        )
        .properties(height=250)
    )
    st.altair_chart(snapshot_chart, use_container_width=True)

    img = Image.open("img/spaceship.jpg")
    st.image(img)
