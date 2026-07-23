
# The Commercial Space Boom: Predicting Launch Growth and Success in the Modern Space Industry

[Description]

The data used for this analysis has been taken from the [Launch Library 2](https://thespacedevs.com/llapi), using their Launches dataset V.2.3.0.

# Question & Scope

<span style="color: #8FF1FA">*How have geography, launch provider, rocket type, and mission characteristics shaped global launch growth since 2010, and how well can those patterns predict future launch volume and launch success?*</span>


The sub-questions taken from the larger question:

1. **Time Series**: How has the number of launches changed each year from 2010-2026, and what does that trend suggest about future launch outcomes?
2. **Geography**: Which countries and launch sites have grown the most, and what does the spread of launches look like across the global horizon?
3. **Launch Success**: Which factors are associated with launch success? (Provider, rocket, country, mission type, orbit, year, etc.)
4. **Prediction**: Can past launch trends predict the number of future launches? Can launch characteristics predict whether a launch succeeds?



___

# Technical Overview

The following an overview of the technical analysis that was done for this project:
1. Data cleanup
2. Time analysis: plotting how the launch environment has changed over time and creating future predictions.
3. Geographical analysis: plotting where most launches occur in the world and tracking any patterns of launches across the global horizon.
4. Base statistics: launches over time, launches by country, launches by status, etc.
5. Looking at success rates, orbits, and provider breakdowns.
6. Predicition Model: building a logistic regression model to predict the success status of a launch.
7. Looking at feature impacts from the model

___

Please use app.py on Streamlit to view the analysis presentation.