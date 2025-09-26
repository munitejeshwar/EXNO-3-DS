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
    df=pd.read_csv("data.csv")
     df
    
<img width="473" height="391" alt="image" src="https://github.com/user-attachments/assets/017827f4-d287-413b-b93e-a0e3e393b8d7" />


    
    from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
    df1 = df.copy()
    education=["High School","Diploma","Bachelors","Masters","PhD"]
    enc=OrdinalEncoder(categories=[education])
    enc.fit_transform(df1[['Ord_2']])
    

<img width="165" height="204" alt="image" src="https://github.com/user-attachments/assets/f18e8954-197e-4812-a421-6599f8a77721" />



    
    df1['OrdinalEncoder']=enc.fit_transform(df1[['Ord_2']])
    df1
    

<img width="617" height="394" alt="image" src="https://github.com/user-attachments/assets/b8b0fbb9-94cf-4b1c-9f27-a1f2b773d2a6" />


    df2 = df.copy()
    enc=LabelEncoder()
    df2['LabelEncoder']=enc.fit_transform(df1[['Ord_2']])
    df2

 <img width="583" height="391" alt="image" src="https://github.com/user-attachments/assets/91cd6e06-9ecf-4ad2-9955-6564ed5623d4" />

    
    from sklearn.preprocessing import OneHotEncoder
    df3=df.copy()
    enc=OneHotEncoder()
    newdata=pd.DataFrame(enc.fit_transform(df3[['City']]))
    df4=pd.concat([df3,newdata],axis=1)
    df4


<img width="545" height="403" alt="image" src="https://github.com/user-attachments/assets/68edf923-ac39-40bd-8039-ac4abc8d5ada" />

    
    pd.get_dummies(df4,columns=["City"])    

 <img width="881" height="392" alt="image" src="https://github.com/user-attachments/assets/4a5f0300-b0b7-49bb-b6d8-8fee6ef45881" />

    
    pip install --upgrade category_encoders
    

<img width="1239" height="298" alt="image" src="https://github.com/user-attachments/assets/2f42635a-da06-47a8-aa79-c9612ab52168" />

    
    from category_encoders import BinaryEncoder
    df5=df.copy()
    enc=BinaryEncoder()
    newdata=pd.DataFrame(enc.fit_transform(df5[['Ord_1']]))
    df6=pd.concat([df5,newdata],axis=1)
    df6
    

<img width="677" height="389" alt="image" src="https://github.com/user-attachments/assets/5b562a13-61ca-45b2-9715-01ccc8616a3b" />

    
    from category_encoders import TargetEncoder
    df7=df.copy()
    enc=TargetEncoder()
    newdata=pd.DataFrame(enc.fit_transform(df7[['Ord_1']],df7['Target']))
    df8=pd.concat([df7,newdata],axis=1)
    df8

<img width="214" height="221" alt="image" src="https://github.com/user-attachments/assets/583bbbb7-c96c-4581-ac84-c69107edcdab" />

    
    import pandas as pd
    df = pd.read_csv("Data_to_Transform.csv")
    df
    

<img width="757" height="468" alt="image" src="https://github.com/user-attachments/assets/52489b0a-2f10-45ab-8d68-385e18926c0f" />

    
    df.skew()
    


    
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import numpy as np
    sm.qqplot(df["Moderate Positive Skew"],line="45")
    plt.show

<img width="663" height="528" alt="image" src="https://github.com/user-attachments/assets/56f38e65-b1a2-46e6-9a37-e81c752d87b1" />

    
    sm.qqplot(df["Highly Positive Skew"],line="45")
    plt.show    

<img width="665" height="529" alt="image" src="https://github.com/user-attachments/assets/430ee4ef-909b-4c0a-81c9-8ab4c5931b25" />

    
    sm.qqplot(df["Moderate Negative Skew"],line="45")
    plt.show
    

<img width="678" height="525" alt="image" src="https://github.com/user-attachments/assets/38907e38-87c0-461f-a253-6318d4a0afae" />

    
    sm.qqplot(df["Highly Negative Skew"],line="45")
    plt.show
    

<img width="642" height="524" alt="image" src="https://github.com/user-attachments/assets/2a2e02d1-0c0f-41d3-9c35-4827858e8373" />

    
    sm.qqplot(df["Highly Negative Skew"],line="45")
    plt.show
    

<img width="652" height="520" alt="image" src="https://github.com/user-attachments/assets/2f0c670b-2577-41c6-a121-cf6a32c793cc" />

    
    df1=df.copy()
    df1['log transformation']=np.log(df["Moderate Positive Skew"])
    df1
    
<img width="1331" height="358" alt="image" src="https://github.com/user-attachments/assets/8b713e48-8cd7-4978-9dca-2375edbba8da" />

    
    sm.qqplot(df1["log transformation"],line="45")
    plt.show
    
<img width="665" height="529" alt="image" src="https://github.com/user-attachments/assets/c3900627-6905-4d09-8739-5f4fe2f13fac" />

    
    df1=df.copy()
    df1['log transformation']=np.log(df["Highly Positive Skew"])
    df1

<img width="957" height="469" alt="image" src="https://github.com/user-attachments/assets/1e2a6819-ef68-4d21-ac58-ffc5ff256d3a" />

    
    sm.qqplot(df1["log transformation"],line="45")
    plt.show    

<img width="657" height="527" alt="image" src="https://github.com/user-attachments/assets/f12c0715-8526-4222-83b0-94cb97218592" />

    df2=df1.copy()
    df2['square root transformation']=np.sqrt(df1["Moderate Positive Skew"])
    df2

<img width="1144" height="499" alt="image" src="https://github.com/user-attachments/assets/220f2cc0-d1ba-410f-ae51-3ec3ed74cc80" />


    sm.qqplot(df2['square root transformation'],line="45")
    plt.show()

<img width="760" height="614" alt="image" src="https://github.com/user-attachments/assets/8f407010-b9e0-43db-a138-29fbfb83838b" />

    df3=df2.copy()
    df3['square transformation']=np.square(df2["Highly Positive Skew"])
    df3

<img width="1258" height="554" alt="image" src="https://github.com/user-attachments/assets/cceefd25-5d81-41f5-89c2-e4706b991c03" />

    sm.qqplot(df3['square transformation'],line="45")
    plt.show()

<img width="728" height="618" alt="image" src="https://github.com/user-attachments/assets/418b234b-50f8-47a0-ac7f-c0d7d9e476fb" />


    df3=df2.copy()
    df3['square transformation']=np.square(df2["Moderate Positive Skew"])
    df3

<img width="1256" height="524" alt="image" src="https://github.com/user-attachments/assets/2ff6f0b2-2788-4d5f-9076-1d945507f157" />


    sm.qqplot(df3['square transformation'],line="45")
    plt.show()

<img width="849" height="612" alt="image" src="https://github.com/user-attachments/assets/9793ce70-391e-455d-ad30-fe5a1733ddcb" />


    df4=df.copy()
    df4['reciprocal transformation']=1/(df4["Moderate Positive Skew"])
    df4

<img width="1126" height="503" alt="image" src="https://github.com/user-attachments/assets/983eaff8-d9ad-487e-8f64-500bff693192" />


    sm.qqplot(df4['reciprocal transformation'],line="45")
    plt.show()

<img width="789" height="626" alt="image" src="https://github.com/user-attachments/assets/f1a5a3ac-0e33-43c6-b1a3-4b48c10f8526" />


    df5=df.copy()
    df5['boxcox transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
    df5

<img width="1178" height="552" alt="image" src="https://github.com/user-attachments/assets/1f237d49-9c2e-4523-b462-5f9147af1e77" />


    sm.qqplot(df5['boxcox transformation'],line="45")
    plt.show()

<img width="779" height="609" alt="image" src="https://github.com/user-attachments/assets/f5e6f662-5c36-4157-bfd7-80fa37cd6646" />


    df6=df.copy()
    df6['yeojohnson transformation'],p=stats.yeojohnson(df6["Moderate Negative Skew"])
    df6

<img width="1036" height="497" alt="image" src="https://github.com/user-attachments/assets/33cb8ff0-7d6c-4a14-85d7-97d09182c13b" />


    sm.qqplot(df6['yeojohnson transformation'],line="45")
    plt.show()

<img width="789" height="607" alt="image" src="https://github.com/user-attachments/assets/f95aa149-aaab-4491-8355-def04a858b17" />


    from sklearn.preprocessing import QuantileTransformer
    df7=df.copy()
    qt=QuantileTransformer(output_distribution='normal')
    df7['Quantile Transformation']=qt.fit_transform(df7[['Highly Positive Skew']])
    df7

<img width="1021" height="570" alt="image" src="https://github.com/user-attachments/assets/50d772e6-f0cb-4304-9736-045b2be00b28" />

```
sm.qqplot(df7['Quantile Transformation'],line="45")
plt.show()
```
<img width="724" height="606" alt="image" src="https://github.com/user-attachments/assets/d310f219-0543-433e-9b3e-11afbeabf358" />

  # RESULT:
  Hence performing Feature Encoding and Transformation process is Successful.
