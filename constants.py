

# Inputs
IN_DATA_PATH = r"./data/bank_data.csv"

# Outputs
OUT_EDA_PATH = r"./images/eda"
OUT_TESTS_EDA_PATH = r"./images/tests/eda/"
OUT_MODEL_REPORTS_PATH = r"./images/model_report/"
OUT_TESTS_MODEL_REPORTS_PATH = r"./images/tests/model_report/"
OUT_MODEL_ARTIFACTS_PATH = r"./models/"
OUT_TESTS_MODEL_ARTIFACTS_PATH = r"./models/tests"

# ID Columns
ID_COLUMNS_LST = ['CLIENTNUM']

# Target Column
TARGET_IN_COLUMN = "Attrition_Flag"
TARGET_OUT_COLUMN = "churn"
TARGET_FUNCTION = lambda val: 0 if val == "Existing Customer" else 1

# Train / Test Split Configs
TEST_SIZE= 0.30
RANDOM_STATE= 76749
