# credit-risk-model
Comparative study of 3 ML models to enhance credit risk assessment

README_COMP1884_group5.txt
=======================

Project Title:
---------------
Leveraging Machine Learning Algorithms to Enhance Credit Risk Assessment in the Financial Sector

Group:
-------
Group 5 - MSc Data Science and its Applications

Contributors:
-------------
- Luc: Random Forest Classifier
- Long: Multi-layer Perceptron Neural Network
- Caleb: Logistic Regression

Data Source:
------------
- Lending Club Loan Data from Kaggle
- Dataset URL: https://www.kaggle.com/datasets/joebeachcapital/lending-club/data
- Description: Contains data for 10,000 individual loan applicants with 55 features (36 numeric, 18 categorical, 1 text)
- Note: The dataset is freely accessible upon logging into a Kaggle account.

Summary Statistics:
-------------------
- Observations: 10,000
- Features: 55
- Missing Values: 42,191 (≈7.7%)
- Duplicate Records: None

Representative Sample:
-----------------------
A snapshot of the dataset includes:
- Personal/financial details: `annual_income`, `emp_length`, `debt_to_income`
- Loan details: `loan_amount`, `interest_rate`, `term`
- Credit history: `delinq_2y`, `inquiries_last_12m`, `public_record_bankrupt`

Libraries Used:
---------------

Initial Implementation (Group_LC_initial_implementation.py):
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import hashlib
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random

Enhanced Implementation (Group_LC_enhanced_implementation.py):
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import hashlib
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

Instructions to Access/Reproduce Data:
--------------------------------------
1. Visit: https://www.kaggle.com/datasets/joebeachcapital/lending-club/data
2. Create a Kaggle account if not already registered.
3. Click “Download” to obtain the CSV files for local use.
4. Use `pandas.read_csv()` in Python to load the data.

Example:
import pandas as pd
df = pd.read_csv("loans_full_schema.csv")

Exploratory Data Analysis (EDA) Report:
---------------------------------------
The following code generates an automated exploratory data analysis (EDA) report using the `ydata_profiling` library (formerly `pandas_profiling`):

```python
# Create initial report
profile = ProfileReport(lending_club, title="CS Lending Club Report")
profile.to_file("CS Lending Club Report.html")

Note:
-----
Ensure `loans_full_schema.csv` is placed in the working directory or provide the full file path when loading it.
