import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# merge the dataset 1 using the key_left key to the dataset 2 using the key_right
def merge_dataset(dataset1: pd.DataFrame, dataset2: pd.DataFrame, key_left: str, key_right: str):

    return pd.merge(dataset1, dataset2, left_on=key_left, right_on=key_right)



# Function to map a date to a season
def map_to_season(date_str):
    month, day = map(int, date_str.split('-'))
    
    # Spring
    if 3 <= month <= 5:
        return 1
    # Summer
    elif 6 <= month <= 8:
        return 2
    # Autumn
    elif 9 <= month <= 11:
        return 3
    # Winter
    else:
        return 4
    


# function for histplot paired with boxplot
def box_hist_plot(columns_to_select, dataset, n_bins, kde=True):
    # Crea un grafico per ogni colonna
    for col in columns_to_select:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Crea una figura con due subplot fianco a fianco
        
        # Histplot
        sns.histplot(dataset[col], bins=n_bins, kde=kde, ax=axes[0], edgecolor='black')
        axes[0].set_title(f'Histplot of {col}')
        axes[0].set_xlabel('Values')
        axes[0].set_ylabel('Frequency')
        
        # Box plot
        sns.boxplot(y=dataset[col], ax=axes[1])
        axes[1].set_title(f'Box Plot of {col}')
        axes[1].set_xlabel('Values')
        
        # Mostra il grafico
        plt.tight_layout()
        plt.show()