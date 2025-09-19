## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
    
     import pandas as pd
     df=pd.read_csv("/content/Encoding Data.csv")
     df
    
  ![img1](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/eca0e23d-6ea3-4685-ad3e-3a3c688afce4)


    
    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
    pm=['Hot','Warm','Cold']
    e1=OrdinalEncoder(categories=[pm])
    e1.fit_transform(df[["ord_2"]])
    

  ![img 2](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/6f0fce59-7852-489d-a76f-db5988a45a3b)



    
    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
    

  ![img 3](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/84e9360b-9728-444d-bbe3-f7480e9633f6)

    
    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
    
  ![img 4](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/addbdb92-ff8a-41f3-af9e-bd97ac6800a2)

    
    le=LabelEncoder()
    dfc=df.copy()
    dfc['ord_2']=le.fit_transform(dfc['ord_2'])
    dfc
    

  ![img 5](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/1c7de496-371e-4a21-a189-7ed70ecc2900)

    
    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder(sparse=False)
    df2=df.copy()
    enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
    df2=pd.concat([df2,enc],axis=1)
    df2
    

  ![img 6](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/a8c5038b-2814-4b1d-8f85-c27b292c04d4)

    
    pd.get_dummies(df2,columns=["nom_0"])
    

  ![img 7](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/797cb3cf-cc31-4c39-ba12-af1d740bbbed)

    
    pip install --upgrade category_encoders
    

  ![img 8](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/eba5c171-3e23-483f-b4ed-6f4c47b5e89b)

    
    from category_encoders import BinaryEncoder
    df=pd.read_csv("/content/data.csv")
    be=BinaryEncoder()
    nd=be.fit_transform(df['Ord_2'])
    fb=pd.concat([df,nd],axis=1)
    dfb1=df.copy()
    dfb
    

  ![img 9](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/9b983cb5-c712-49d8-bcd8-817ca7f56947)

    
    from category_encoders import TargetEncoder
    te=TargetEncoder()
    cc=df.copy()
    new=te.fit_transform(X=cc["City"],y=cc["Target"])
    cc=pd.concat([cc,new],axis=1)
    cc
    

  ![i 0](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/0d2e7e51-9cb4-4530-a56d-39d1435d0249)

    
    import pandas as pd
    from scipy import stats
    import numpy as np
    df=pd.read_csv("/content/Data_to_Transform.csv")
    df
    

  ![i 1](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/f71a4dd5-6389-4fb9-a209-81064b1d878f)

    
    df.skew()
    

  ![i 2](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/3e30db2a-11cd-4980-a53d-87e495cdba2d)

    
    np.log(df["Highly Positive Skew"])
    

  ![i 3](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/ea34c360-81a5-4760-bc9e-2c6a9d07be6c)

    
    np.reciprocal(df["Moderate Positive Skew"])
    

  ![i 4](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/790eadf6-e770-4874-943c-ddfb7bca5113)

    
    np.sqrt(df["Highly Positive Skew"])
    

  ![i 5](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/7e2ab48c-7321-477e-81e4-ba3d6b132bf5)

    
    np.square(df["Highly Positive Skew"])
    

  ![i 6](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/bcda767c-4193-4548-89ae-8ba49be8a323)

    
   df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
   df
    

  ![i 7](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/7632047a-430d-4b18-8bab-abc86a475a5a)

    
    df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
    df.skew()
    
  ![i 8](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/90edf1a6-0480-49e3-9fe6-7ac276d2acf3)

    
    df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
    df.skew()
    
  ![i 9](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/af78ba87-bd09-43e8-a612-406cad9d7686)

    
   import matplotlib.pyplot as plt
   import seaborn as sns
   import statsmodels.api as sm
   import scipy.stats as stats

   sm.qqplot(df["Moderate Negative Skew"],line='45')

   plt.show()
    

  ![m1](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/b72a5319-b492-4c91-981e-15bf8d38c539)

    
    sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
    

  ![m2](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/9c011486-2fd5-4752-b434-702ecc5bebdc)

    
    from sklearn.preprocessing import QuantileTransformer
    qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

    df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()
    
    

  ![m 3](https://github.com/nanditha121/EXNO-3-DS/assets/142209508/880ace95-7dd1-4732-941d-e007439a6fc5)


  # RESULT:
        Hence performing Feature Encoding and Transformation process is Successful.
