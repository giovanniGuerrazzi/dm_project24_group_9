from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
import pandas
import numpy as np


# Function to normalize dataset
def __transform_single_features(dataset: pandas.DataFrame, transformation: str) -> Tuple[
    pandas.DataFrame, Dict[str, Any]]:
    match transformation:
        case "standard":
            transformed_dataset = dataset.copy().select_dtypes(exclude=["object", "category", "bool", "datetime64"])
            transformations = dict()

            for feature in transformed_dataset.columns:
                transformations[feature] = StandardScaler()
                transformed_feature = transformations[feature].fit_transform(transformed_dataset[[feature]]).squeeze()
                transformed_dataset = transformed_dataset.astype({feature: transformed_feature.dtype})
                transformed_dataset.loc[:, feature] = transformed_feature
        case _:
            raise ValueError(f"Unknown transformation: {transformation}")

    return transformed_dataset, transformations


def center_and_scale(dataset: pandas.DataFrame) -> Tuple[pandas.DataFrame, Dict[str, Any]]:
    """Shifts data to the origin: removes mean and scales by standard deviation all numeric features. Returns a copy of the dataset."""
    return __transform_single_features(dataset, "standard")


# Function to calculate bins dynamically
def calculate_bins(data, method='sturges'):
    if method == 'sturges':
        return int(np.ceil(1 + np.log2(len(data))))
    elif method == 'sqrt':
        return int(np.ceil(np.sqrt(len(data))))
    elif method == 'fd':  # Freedman-Diaconis Rule
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1/3))
        return int(np.ceil((data.max() - data.min()) / bin_width))
    else:
        raise ValueError("Invalid method. Choose 'sturges', 'sqrt', or 'fd'.")