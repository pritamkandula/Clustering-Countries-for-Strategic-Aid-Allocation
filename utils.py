from scipy.stats import boxcox
import pickle

# Load pickled models and metadata
with open("kmeans_model.pkl", "rb") as pickle_kmeans:
    kmeans_model = pickle.load(pickle_kmeans)

with open("scaler.pkl", "rb") as pickle_scaler:
    scaler = pickle.load(pickle_scaler)

with open("outlier_bounds.pkl", "rb") as pickle_outlier_bounds:
    outlier_bounds_dict = pickle.load(pickle_outlier_bounds)

with open("lambdas.pkl", "rb") as pickle_lambdas:
    lambdas = pickle.load(pickle_lambdas)


def outlier_check(user_data, bounds_dict):
    """
    Checks if user_data has outlier values based on bounds_dict.
    Raises a ValueError if any feature value is outside the allowed range.
    :param user_data: dict, raw input data obtained from the user.
    :param bounds_dict: dict, upper and lower bounds for each feature.
    """
    for feature, bounds in bounds_dict.items():
        if feature not in user_data:
            raise KeyError(f"Feature {feature} is missing from user data.")

        lower = bounds['lower_bound']
        upper = bounds['upper_bound']

        if user_data[feature] < lower or user_data[feature] > upper:
            raise ValueError(
                f"The value for {feature} is an outlier. It must lie within the range of {lower} and {upper}."
            )
    return user_data


def positive_value_converter(user_data):
    """
    Converts negative values in user_data to positive values by applying a shift.
    If all values are positive, returns user_data unchanged.
    :param user_data: dict, raw input data obtained from the user.
    """
    min_value = min(user_data.values())
    if min_value < 0:
        shift_value = abs(min_value) + 1  # Calculate shift value to make all values positive
        user_data = {feature: value + shift_value for feature, value in user_data.items()}
    return user_data


def boxcox_transform(user_data):
    """
    Applies boxcox transformation to gdpp, child_mort, and inflation features to avoid skewness.
    Retains other features in the final output unchanged.
    :param user_data: dict, raw input data obtained from the user.
    """
    processed_data = user_data.copy()
    for feature in ['gdpp', 'child_mort', 'inflation']:
        if feature in user_data:
            processed_data[feature] = boxcox(user_data[feature] + 1, lambdas[feature])  # Add 1 to handle zeros
        else:
            raise KeyError(f"Feature {feature} is missing from the user data.")

    return processed_data
