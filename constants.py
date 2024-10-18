

# Inputs
IN_DATA_PATH = r"./data/bank_data.csv"

# Outputs
OUT_EDA_PATH = r"./images/eda"
OUT_TESTS_EDA_PATH = r"./images/tests/eda/"

# Target Column
TARGET_IN_COLUMN = "Attrition_Flag"
TARGET_OUT_COLUMN = "churn"
TARGET_FUNCTION = lambda val: 0 if val == "Existing Customer" else 1

# Features

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]
