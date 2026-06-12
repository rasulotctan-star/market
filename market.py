# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load dataset
df = pd.read_csv('market.csv')

# Remove identifier columns that do not contribute to prediction
df.drop(columns=['Item_Identifier', 'Outlet_Identifier'], inplace=True)

# Fill missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

# Encode categorical variables into numerical values
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separate features and target variable
x = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create and train XGBoost regression model
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03
)

model = xgb.fit(x_train, y_train)

# Save trained model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)