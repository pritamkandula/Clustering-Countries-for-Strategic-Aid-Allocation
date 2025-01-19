from utils import outlier_check, positive_value_converter, boxcox_transform, \
                outlier_bounds_dict, scaler, kmeans_model

from flask import Flask, request, jsonify

import pandas as pd

app = Flask(__name__)


@app.route("/", methods=['GET'])
def title():
    """
    Returns the title of the Project model
    """
    return "<h1> Country Status Analyzer </h1>"


@app.route("/predict", methods=['POST'])
def predict_label():
    """
    Returns the prediction made by the model based on the input country data
    label 0 : Highly Developed Country
    label 1: Least Developed Country
    label 2: Moderately Developed Country
    """
    try:
        country_data = request.get_json()
        if not country_data:
            return jsonify({"error": "Invalid input. JSON data is required."}), 400

        # Ensure all required features are present
        required_features = outlier_bounds_dict.keys()
        missing_features = [feature for feature in required_features if feature not in country_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        # Step 1: Check for outliers
        outlier_free_country_data = outlier_check(country_data, outlier_bounds_dict)

        # Step 2: Ensure positive values
        positive_country_data = positive_value_converter(outlier_free_country_data)

        # Step 3: Apply Box-Cox transformation
        transformed_country_data = boxcox_transform(positive_country_data)

        # Step 4: Scale data and Ensure that the scaler uses the same feature names as during training
        transformed_country_data_for_prediction = pd.DataFrame({
            'exports': [transformed_country_data['exports']],
            'health': [transformed_country_data['health']],
            'gdpp_boxcox': [transformed_country_data['gdpp']],
            'child_mort_boxcox': [transformed_country_data['child_mort']],
            'inflation_boxcox': [transformed_country_data['inflation']]
        })
        scaled_country_data = scaler.transform(transformed_country_data_for_prediction)

        # Step 5: Predict the label
        scaled_country_data_df = pd.DataFrame(scaled_country_data, columns=[
            'exports', 'health', 'gdpp_boxcox', 'child_mort_boxcox', 'inflation_boxcox'])
        result = int(kmeans_model.predict(scaled_country_data_df)[0])

        # Step 6: Map result to message
        label_mapping = {
            0: "Highly Developed Country",
            1: "Least Developed Country",
            2: "Moderately Developed Country"
        }
        message = label_mapping.get(result, "Sorry! Unable to predict the result for this country.")

        return jsonify({"label": result, "message": message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
