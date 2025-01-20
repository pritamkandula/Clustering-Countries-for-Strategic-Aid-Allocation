from utils import kmeans_model, scaler, outlier_bounds_dict, \
        outlier_check, positive_value_converter, boxcox_transform

import streamlit as st
import pandas as pd

country_data_file = "Country-data.csv"

# Streamlit UI
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

# Input Form
st.sidebar.header("Input Country Data")
exports = st.sidebar.number_input("Exports (%)", min_value=-17.0, max_value=100.0, step=0.1)
health = st.sidebar.number_input("Health (% of GDP)", min_value=0.0, max_value=15.0, step=0.1)
gdpp = st.sidebar.number_input("GDP per capita (in USD)", min_value=-17750.0, step=34000.0)
child_mort = st.sidebar.number_input("Child Mortality (per 1000 births)", min_value=0.0, step=150.0)
inflation = st.sidebar.number_input("Inflation (%)", min_value=-10.0, max_value=25.0, step=0.1)

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
    # Handle other chart cases (e.g., numeric vs numeric)
    elif pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
        st.line_chart(df[[x_axis, y_axis]].dropna())
    else:
        st.error("Invalid selection, please choose a numerical Y-axis for the bar chart.")


# Embed Tableau visualization
st.markdown("""
**Explore More**:
Check out the [Tableau Dashboard](https://public.tableau.com/app/profile/pritam.kandula/viz/Clustering_Countries_portfolio/GlobalHealthandEconomicDisparities)  
for a deeper dive into global health and economic disparities.
""")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Pritam Kandula")
