
# coding: utf-8

# In[1]:


import os
os.chdir('E:\ProjectPython')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:


df=pd.read_csv('XYZCorp_LendingData.txt',sep='\t')


# In[3]:


##Scrutiny of the Dataset to understand the number of observations and number of variables
df.info()
##There are 855969 observations and 73 columns(features including our target variable)
##We can also see that there are a lot of missing values in the Dataset


# In[4]:


df.head(2)


# In[5]:


data_types=df.dtypes.value_counts()
data_types
##Out of 73, 52 are numeric variables and 21 are objects 


# In[6]:


##Finding out the missing values and the percentage.
df_missing=df.isnull().sum().reset_index()
df_missing.columns=['Col_Name','Num_of_MV']
df_missing=df_missing[df_missing['Num_of_MV']>0]
df_missing=df_missing.sort_values(by='Num_of_MV',ascending=False)
df_missing['Percentage']=(df_missing['Num_of_MV']/len(df))*100


# In[7]:


df_missing.head(20)
##There are 31 columns with null values and out of which 20 columns have null values greater than 75%. So we can 
##drop these 20 columns since filling null values for these columns doesn't make any sense and it may lead to an
##unbalanced model.


# In[8]:


##Dropping the columns with more than 50% missing values
df_missing=df_missing.reset_index()
Max_Missing=df_missing.iloc[0:21,1].values
df=df.drop(Max_Missing,axis=1)


# In[9]:


##Now lets fill the missing values for the remaining variables
#Filling the missing values
df['title']=df['title'].fillna('Others')
df['tot_coll_amt']=df['tot_coll_amt'].fillna(np.mean(df['tot_coll_amt']))
df['tot_cur_bal']=df['tot_cur_bal'].fillna(np.mean(df['tot_cur_bal']))
df['total_rev_hi_lim']=df['total_rev_hi_lim'].fillna(np.mean(df['total_rev_hi_lim']))
df['collections_12_mths_ex_med']=df['collections_12_mths_ex_med'].fillna(0)
df['revol_util']=df['revol_util'].fillna(np.mean(df['revol_util']))
##Removing rows with missing values
df=df.dropna(axis=0,subset=['emp_title','last_credit_pull_d','last_pymnt_d'])
##For filling the employment with the mode, we have to convert the variable to numeric first and then fill with the mode
df['emp_length']= df['emp_length'].astype(str).str.replace('\D+', '')
df['emp_length']=pd.to_numeric(df.emp_length)
df['emp_length']=df['emp_length'].fillna(10)


# In[10]:


##All the null values have been dealt with. Lets deal with all the variables with datatype-object
#1.Removing months from term variable and converting the variable to a numeric
df['term']=df['term'].apply(lambda x:x.split()[0])
df['term']=pd.to_numeric(df['term'])
#2.Converting the variables with 2 types of observations to numeric
df['initial_list_status']=df['initial_list_status'].apply(lambda x: 1 if "f" in x else 0)
df['application_type']=df['application_type'].apply(lambda x: 1 if "INDIVIDUAL" in x else 0)
df['pymnt_plan']=df['pymnt_plan'].apply(lambda x: 1 if "n" in x else 0)


# In[11]:


##Plotting a heat map to check the correlation between the variables.
plt.figure(figsize=(14,9))
sns.heatmap(df.corr(), annot=True, linewidths=.5,fmt='.2f')
plt.title('Correlation Heat Map')


# In[12]:


##The above plot size can be increased by increasing the figsize to 16,10
#There are few variables with correlation.
#Removing the ones with correlation to avoid multicollinearity. Keeping the variables having high correlation with the 
#target variable
##Also the variable like id, member_id and payment plan are unique.
##The variable employment title has 288930 unique values [df['emp_title'].nunique()]. This is a high cardinality variable
## and can be removed.
df=df.drop(['id','member_id','policy_code','emp_title','loan_amnt','funded_amnt','installment','total_rec_prncp','out_prncp',
            'total_pymnt','total_rev_hi_lim','collection_recovery_fee'],axis=1)


# In[13]:


##Checking the dataset once again to have an idea about the null values and the datatypes
df.info()


# In[14]:


##There is one more column with a large number of null values. Dropping the variable "Next payment date" since filling 
##the null values may distort our model.
df=df.drop(['next_pymnt_d'],axis=1)


# In[15]:


##Doing some plotting to check the target variable and some categorical variables
df['default_ind'].hist(bins=20)
##The histogram clearly shows that the number of non defaulters are far more than the number of defaulters.
##The Dataset is an imbalanced dataset. I will deal with the imbalance later.


# In[16]:


df['grade'].hist(bins=20)
##The plot shows that the majority of the customers have A,B or C ratings. I will deal with the grade later
##by extracting a new feature.


# In[17]:


##Creating a box plot for the funded amount to check for the outliers.
df.boxplot(column='funded_amnt_inv')
##There are no outliers.


# In[18]:


df['home_ownership'].hist(bins=20)
##Majority of the customer either have mortgaged their houses or are living in rented houses


# In[19]:


##Dealing with object datatypes
##Grade and sub-grade are related to each other. We are keeping grade and removing sub-grade due to cardinality issues
grade_list=['A','B','C']
df['grade']=df['grade'].apply(lambda x: 1 if x in grade_list else 0)
##Coverting the home_ownership variable
df['home_ownership']=df['home_ownership'].apply(lambda x: 1 if "MORTGAGE" in x else 0)
##The variable purpose and title are more or less same. Hence we can remove title from our DS.
df['purpose']=df['purpose'].apply(lambda p: 1 if "debt_consolidation" in p else 0)
##Verification status
ver_stat=['Source Verified','Verified']
df['verification_status']=df['verification_status'].apply(lambda x: 1 if x in ver_stat else 0)


# In[20]:


##FEATURE ENGINEERING
#Creating number of days out of last payment date- New Feature 1
df['last_pymnt_d']=pd.to_datetime(df['last_pymnt_d'])
current_time=df.last_pymnt_d.max()
current_time
df['Time_diff_def']=df.last_pymnt_d.map(lambda x: current_time-x) 
df['Time_diff_days_def']=df.Time_diff_def.map(lambda x: x.days)

#Creating number of days out of existing credit line date- New Feature 2
df['earliest_cr_line']=pd.to_datetime(df['earliest_cr_line'])
current_time=df.earliest_cr_line.max()
current_time
df['Time_diff']=df.earliest_cr_line.map(lambda x: current_time-x) 
df['Time_diff_days_cr']=df.Time_diff.map(lambda x: x.days)

#Creating a new variable for zip code- New feature 3
df['zip_code']=df['zip_code'].apply(lambda x: x[0:3])
df['zip_code']=pd.to_numeric(df['zip_code'])
df['Norm_Zip']=(df['zip_code']/len(df))*100


# In[21]:


##Removing some more features after feature engineering and datatype conversion
df=df.drop(['sub_grade','zip_code','addr_state','title','last_pymnt_d','earliest_cr_line','last_credit_pull_d',
            'Time_diff_def','Time_diff'],axis=1)


# In[22]:


##Checking the dataset again
df.info()


# In[23]:


##Splitting the dataset into train and test as per the requirement
##training set-> Observations upto June 2015
##testing set-> Observations from June 2015 to December 2015
df['issue_d']=pd.to_datetime(df['issue_d'])
train_data=df[df['issue_d']<'2015-06-01']
test_data=df[df['issue_d']>='2015-06-01']


# In[24]:


##Dropping the issue date from training data and testing data
train_data=train_data.drop('issue_d',axis=1)
test_data=test_data.drop('issue_d',axis=1)

##Further splitting the train data.
X=train_data.drop('default_ind',axis=1)
y=train_data['default_ind']

##Splitting the test data
test_X=test_data.drop('default_ind',axis=1)
test_y=test_data['default_ind']


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[28]:


##Trying on our original test set
pred=LR.predict(test_X)
print(classification_report(test_y,pred))
print(confusion_matrix(test_y,pred))


# In[30]:


##The model seems to be an overfitted model.The model is giving excellent results in training data but not good result in
##test data. This may be because of the imbalanced dataset.
#I tried undersampling method to remove the imbalance.
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)
X_resampled1, y_resampled1, idx_resampled1 = rus.fit_sample(test_X, test_y)


# In[32]:


LR.fit(X_resampled,y_resampled)
pred_y=LR.predict(X_resampled1)
print(classification_report(y_resampled1,pred_y))
print(confusion_matrix(y_resampled1,pred_y))


# In[33]:


##Model 2 with Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_resampled,y_resampled)
y_pred1=rfc.predict(X_resampled1)
print(classification_report(y_resampled1,y_pred1))
print(confusion_matrix(y_resampled1,y_pred1))


# In[34]:


##The model is giving better results with undersampling.
##I have also tried Principal Component Analysis to do some dimensionality reduction.
##For PCA the features needs to be normalised using Standard Scaler.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sc=StandardScaler()
X_resampled=sc.fit_transform(X_resampled)
X_resampled1=sc.transform(X_resampled1)
pca=PCA()
X_resampled=pca.fit_transform(X_resampled)
X_resampled1=pca.transform(X_resampled1)
expl_var=pca.explained_variance_ratio_
expl_var=pd.DataFrame(expl_var)


# In[35]:


expl_var.loc[0:16,0].sum()


# In[36]:


##77% of the variance is explained by 16 components.
pca=PCA(n_components=15)
X_resampled=pca.fit_transform(X_resampled)
X_resampled1=pca.transform(X_resampled1)


# In[37]:


LR.fit(X_resampled,y_resampled)
y_pred=LR.predict(X_resampled1)
print(classification_report(y_resampled1,y_pred))
print(confusion_matrix(y_resampled1,y_pred))


# In[38]:


rfc.fit(X_resampled,y_resampled)
y_pred1=rfc.predict(X_resampled1)
print(classification_report(y_resampled1,y_pred1))
print(confusion_matrix(y_resampled1,y_pred1))


# In[ ]:


##WITH RESPECT TO THE TRUE POSITIVES, RANDOM FOREST MODEL IS BETTER AND WITH RESPECT TO TRUE NEGATIVES
#LOGISTIC REGRESSION MODEL IS BEHAVING BETTER.

