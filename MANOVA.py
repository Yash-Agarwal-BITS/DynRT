import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# Load the dataset
try:
    df = pd.read_csv('Sarcasm DynRT IIIT.xlsx - Router_Values.csv')
    print("File 'Sarcasm DynRT IIIT.xlsx - Router_Values.csv' loaded successfully.")
    print("First 5 rows of the dataframe:")
    print(df.head())
    print("\nDataframe Info:")
    df.info()

    # Identify dependent variables (router values) and the independent variable (type)
    # Based on the previous context, the router values start from column 'router_vector_0'
    dependent_vars = df.columns[3:]
    independent_var = 'type'

    # Drop rows with missing values in the relevant columns
    df_cleaned = df.dropna(subset=[independent_var] + list(dependent_vars))

    # The formula for MANOVA
    formula = " + ".join(dependent_vars) + " ~ " + independent_var

    # Perform MANOVA
    manova = MANOVA.from_formula(formula, data=df_cleaned)
    result = manova.mv_test()

    print("\nMANOVA Test Results:")
    print(result)

except FileNotFoundError:
    print("Error: 'Sarcasm DynRT IIIT.xlsx - Router_Values.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")







''' 

    --------- OUTPUT --------
File 'Sarcasm DynRT IIIT.xlsx - Router_Values.csv' loaded successfully.
First 5 rows of the dataframe:
       image_id  label           type  router_vector_0  router_vector_1_0  router_vector_1_1  router_vector_2_0  router_vector_2_1  router_vector_2_2  router_vector_3_0  router_vector_3_1  router_vector_3_2  router_vector_3_3
0  8.204110e+17      0  non sarcastic                1           0.436983           0.563017           0.344889           0.324566           0.330545           0.241756           0.216282           0.240118           0.301844
1  8.200530e+17      0  non sarcastic                1           0.508915           0.491085           0.337836           0.331401           0.330763           0.245487           0.272045           0.247316           0.235152
2  8.233180e+17      0  non sarcastic                1           0.454096           0.545904           0.379946           0.307114           0.312940           0.251735           0.231021           0.254955           0.262289
3  8.200560e+17      0  non sarcastic                1           0.518298           0.481702           0.337966           0.323625           0.338409           0.250473           0.258906           0.251198           0.239424
4  8.182380e+17      0  non sarcastic                1           0.508660           0.491340           0.362780           0.333848           0.303372           0.228303           0.238735           0.268934           0.264028

Dataframe Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 128 entries, 0 to 127
Data columns (total 13 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   image_id           128 non-null    float64
 1   label              128 non-null    int64  
 2   type               128 non-null    object 
 3   router_vector_0    128 non-null    int64  
 4   router_vector_1_0  128 non-null    float64
 5   router_vector_1_1  128 non-null    float64
 6   router_vector_2_0  128 non-null    float64
 7   router_vector_2_1  128 non-null    float64
 8   router_vector_2_2  128 non-null    float64
 9   router_vector_3_0  128 non-null    float64
 10  router_vector_3_1  128 non-null    float64
 11  router_vector_3_2  128 non-null    float64
 12  router_vector_3_3  128 non-null    float64
dtypes: float64(10), int64(2), object(1)
memory usage: 13.1+ KB

MANOVA Test Results:
                                Multivariate linear model
==========================================================================================
                                                                                          
------------------------------------------------------------------------------------------
       Intercept               Value         Num DF  Den DF         F Value         Pr > F
------------------------------------------------------------------------------------------
          Wilks' lambda               0.0000 7.0000 118.0000 15377921002896770.0000 0.0000
         Pillai's trace               1.0127 7.0000 118.0000             -1348.7312 1.0000
 Hotelling-Lawley trace 900719925474098.2500 7.0000 118.0000 15183564457991944.0000 0.0000
    Roy's greatest root 900719925474098.2500 7.0000 118.0000 15183564457991942.0000 0.0000
------------------------------------------------------------------------------------------
                                                                                          
-----------------------------------------------------------------------------------------------
                  type               Value        Num DF       Den DF       F Value      Pr > F
-----------------------------------------------------------------------------------------------
                  Wilks' lambda      0.7288      30.0000      338.2237       1.2832      0.1512
                 Pillai's trace      0.2940      30.0000      351.0000       1.2712      0.1596
         Hotelling-Lawley trace      0.3414      30.0000      254.8848       1.2962      0.1463
            Roy's greatest root      0.2070      10.0000      117.0000       2.4223      0.0117
==========================================================================================

 '''