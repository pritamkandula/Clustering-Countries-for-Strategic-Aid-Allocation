from utils import kmeans_model, scaler, outlier_bounds_dict, \
        outlier_check, positive_value_converter, boxcox_transform

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

country_data_file = "Country-data.csv"

st.set_page_config(page_title="Country Status Analyzer", page_icon="🌍")
st.title("Country Status Analyzer")
st.markdown(
    """
    Welcome to the **Country Status Analyzer**! 🌍

    This application predicts a country's development status based on health and economic indicators:
    - **Highly Developed Country**
    - **Moderately Developed Country**
    - **Least Developed Country**
    """
)

st.sidebar.header("Input Country Data")

exports_slider = st.sidebar.slider("Exports (%)", min_value=-17.0, max_value=100.0,
                                   value=0.0, step=0.1, key="exports_slider")
exports_input = st.sidebar.number_input("Or enter Exports (%)", min_value=-17.0, max_value=100.0,
                                        value=exports_slider, step=0.1, key="exports_input")

health_slider = st.sidebar.slider("Health (% of GDP)", min_value=0.0, max_value=15.0,
                                  value=5.0, step=0.1, key="health_slider")
health_input = st.sidebar.number_input("Or enter Health (% of GDP)", min_value=0.0,
                                       max_value=15.0, value=health_slider, step=0.1, key="health_input")

gdpp_slider = st.sidebar.slider("GDP per capita (in USD)", min_value=-17750.0,
                                max_value=50000.0, value=20000.0, step=500.0, key="gdpp_slider")
gdpp_input = st.sidebar.number_input("Or enter GDP per capita (in USD)", min_value=-17750.0,
                                     max_value=50000.0, value=gdpp_slider, step=500.0, key="gdpp_input")

child_mort_slider = st.sidebar.slider("Child Mortality (per 1000 births)", min_value=0.0,
                                      max_value=150.0, value=50.0, step=1.0, key="child_mort_slider")
child_mort_input = st.sidebar.number_input("Or enter Child Mortality (per 1000 births)", min_value=0.0,
                                           max_value=150.0, value=child_mort_slider, step=1.0, key="child_mort_input")

inflation_slider = st.sidebar.slider("Inflation (%)", min_value=-10.0, max_value=25.0,
                                     value=5.0, step=0.1, key="inflation_slider")
inflation_input = st.sidebar.number_input("Or enter Inflation (%)", min_value=-10.0, max_value=25.0,
                                          value=inflation_slider, step=0.1, key="inflation_input")

# Use the number input values as the final input
exports = exports_input
health = health_input
gdpp = gdpp_input
child_mort = child_mort_input
inflation = inflation_input

if st.sidebar.button("Predict Country Status"):
    try:
        # Step 1: Collect input data
        user_data = {
            "exports": exports,
            "health": health,
            "gdpp": gdpp,
            "child_mort": child_mort,
            "inflation": inflation
        }

        # Step 2: Process data
        user_data = outlier_check(user_data, outlier_bounds_dict)
        user_data = positive_value_converter(user_data)
        transformed_data = boxcox_transform(user_data)

        transformed_df = pd.DataFrame({
            'exports': [transformed_data['exports']],
            'health': [transformed_data['health']],
            'gdpp_boxcox': [transformed_data['gdpp']],
            'child_mort_boxcox': [transformed_data['child_mort']],
            'inflation_boxcox': [transformed_data['inflation']]
        })

        scaled_data = scaler.transform(transformed_df)

        # Step 3: Predict
        scaled_data_df = pd.DataFrame(scaled_data, columns=[
            'exports', 'health', 'gdpp_boxcox', 'child_mort_boxcox', 'inflation_boxcox'])

        prediction = int(kmeans_model.predict(scaled_data)[0])

        label_mapping = {
            0: "Highly Developed Country",
            1: "Least Developed Country",
            2: "Moderately Developed Country"
        }
        st.success(f"Prediction: {label_mapping[prediction]}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


st.subheader("Sample Country Data")
df = pd.read_csv(country_data_file)
# Search functionality
search_query = st.text_input("Search in DataFrame (case-insensitive):")
if search_query:
    filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
    st.markdown(f"### Filtered DataFrame ({len(filtered_df)} rows)")
    st.dataframe(filtered_df)
else:
    st.dataframe(df)


st.markdown("### DataFrame Summary")
st.write(df.describe())

# Data visualization
st.markdown("### Data Visualization")
x_axis = st.selectbox("Select X-axis", options=df.columns)
y_axis = st.selectbox("Select Y-axis", options=df.columns)

# Check if the X-axis is categorical (Country) and Y-axis is numerical
if st.button("Generate Plot"):
    if df[x_axis].dtype == 'object' and pd.api.types.is_numeric_dtype(df[y_axis]):
        # Group by Country for categorical X-axis
        grouped_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        st.bar_chart(grouped_df.set_index(x_axis)[y_axis])
    elif pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
        # Create scatter plot using matplotlib
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.6)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        st.error("Invalid selection. Please select compatible columns.")


# Tableau visualization
st.markdown("""
**Explore More**:
Check out the [Tableau Dashboard](https://public.tableau.com/app/profile/pritam.kandula/viz/Clustering_Countries_portfolio/GlobalHealthandEconomicDisparities)  
for a deeper dive into global health and economic disparities.
""")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Pritam Kandula")
