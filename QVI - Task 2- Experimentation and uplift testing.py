#!/usr/bin/env python
# coding: utf-8

# # Experimentation and uplift testing

# Julia has asked us to evaluate the performance of a store trial which was performed in stores 77, 86 and 88.
# 
# We have chosen to complete this task in R, however you will also find Python to be a useful tool in this piece of analytics. We have also provided an R solution template if you want some assistance in getting through this Task.
# 
# To get started use the QVI_data dataset below or your output from task 1 and consider the monthly sales experience of each store. 
# 
# This can be broken down by:
#     
# 1. total sales revenue
# 2. total number of customers
# 3. average number of transactions per customer
# 
# Create a measure to compare different control stores to each of the trial stores to do this write a function to reduce having to re-do the analysis for each trial store. Consider using Pearson correlations or a metric such as a magnitude distance e.g. 1- (Observed distance – minimum distance)/(Maximum distance – minimum distance) as a measure.
# 
# Once you have selected your control stores, compare each trial and control pair during the trial period. You want to test if total sales are significantly different in the trial period and if so, check if the driver of change is more purchasing customers or more purchases per customers etc.
# 
# Main areas of focus are :
# 
# 1. Select control stores – Explore data, define metrics, visualize graphs
# 2. Assessment of the trial – insights/trends by comparing trial stores with control stores
# 3. Collate findings – summarize and provide recommendations

# ### Import the necessary packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime 
import matplotlib.pyplot as plt
import re


# In[2]:


# Import the QVI_transaction_data dataset into Jupyter Notebooks

file_location = 'C:\\Users\\DELL\\Desktop\\QVI DATA\\'
customer_transaction_data = pd.read_excel(file_location + 'customer_transaction_data1.xlsx')


# In[3]:


# View the first 5 rows of our dataframe

customer_transaction_data.head()


# In[4]:


# View the first 5 rows of our dataframe

customer_transaction_data.tail()


# In[ ]:





# #### Check for nulls

# In[5]:


customer_transaction_data.info()


# In[ ]:





# The client has selected store numbers 77, 86 and 88 as trial stores and want control stores to be established stores that are operational for the entire observation period. We would want to match trial stores to control stores that are similar to the trial store prior to the trial period of Feb 2019 in terms of :
# 
# - Monthly overall sales revenue
# - Monthly number of customers
# - Monthly number of transactions per customer
# 
# Let's first create the metrics of interest and filter to stores that are present throughout the pre-trial period.

# #### Let's add a new month ID, YEARMONTH,column in the data with the format yyyymm

# In[6]:


# create new column YEARMONTH in format yyyymm
customer_transaction_data["DATE"] = pd.to_datetime(customer_transaction_data["DATE"])
customer_transaction_data["YEARMONTH"] = customer_transaction_data["DATE"].dt.strftime("%Y%m").astype("int64")

# View the created MONTH column
customer_transaction_data['YEARMONTH']


# In[ ]:





# #### Create a new data frame measureOverTime to calculate total sales, number of customers, transactions per customer, chips per customer and the average price per unit fore each store and month

# In[7]:


# Group the data by STORE_NBR and YEARMONTH, and calculate the required metrics

def measureOverTime():
    store_YEARMONTH_group = customer_transaction_data.groupby(["STORE_NBR", "YEARMONTH"])
    total_sales = store_YEARMONTH_group["TOT_SALES"].sum()
    num_cust = store_YEARMONTH_group["LYLTY_CARD_NBR"].nunique()
    trans_per_cust = store_YEARMONTH_group.size() / num_cust
    avg_chips_per_cust = store_YEARMONTH_group["PROD_QTY"].sum() / num_cust
    avg_chips_price = total_sales / store_YEARMONTH_group["PROD_QTY"].sum()
    aggregates = [total_sales, num_cust, trans_per_cust, avg_chips_per_cust, avg_chips_price]
    metrics = pd.concat(aggregates, axis = 1)
    metrics.columns = ["TOT_SALES", "nCustomers", "nTxnPerCust", "nChipsPerTxn", "avgPricePerUnit"]
    return metrics
    


# In[8]:


store_monthly_metrics = measureOverTime().reset_index()
store_monthly_metrics.info()


# In[ ]:





# #### Filter to the pre-trial period and stores with full observation periods (12 months) - control stores

# In[9]:


store_full_obs = store_monthly_metrics["STORE_NBR"].value_counts()
full_observ_index = store_full_obs[store_full_obs == 12].index
full_observ = store_monthly_metrics[store_monthly_metrics["STORE_NBR"].isin(full_observ_index)]
preTrialMeasures = full_observ[full_observ["YEARMONTH"] < 201902]

preTrialMeasures.head(10)


# In[10]:


preTrialMeasures.shape


# In[ ]:





# #### Create a function to calculate correlation for a measure, looping through each control store.

# In[11]:


def calculate_correlation(measure, storeComparison, inputTable=preTrialMeasures):
    control_store_nbrs = inputTable[~inputTable["STORE_NBR"].isin([77, 86, 88])]["STORE_NBR"].unique()
    corrs = pd.DataFrame(columns = ["YEARMONTH", "Trial_Str", "Ctrl_Str", "Corr_Score"])
    trial_store = inputTable[inputTable["STORE_NBR"] == storeComparison][measure].reset_index()
    for control in control_store_nbrs:
        concat_df = pd.DataFrame(columns = ["YEARMONTH", "Trial_Str", "Ctrl_Str", "Corr_Score"])
        control_store = inputTable[inputTable["STORE_NBR"] == control][measure].reset_index()
        concat_df["Corr_Score"] = trial_store.corrwith(control_store, axis=1)
        concat_df["Trial_Str"] = storeComparison
        concat_df["Ctrl_Str"] = control
        concat_df["YEARMONTH"] = list(inputTable[inputTable["STORE_NBR"] == storeComparison]["YEARMONTH"])
        corrs = pd.concat([corrs, concat_df])
    return corrs


# In[12]:


corr_table = pd.DataFrame()
for trial_num in [77, 86, 88]:
    corr_table = pd.concat([corr_table, calculate_correlation(["TOT_SALES", "nCustomers", "nTxnPerCust", "nChipsPerTxn", "avgPricePerUnit"], trial_num)])
    
corr_table.head(10)


# In[ ]:





# #### Create a function to calculate a standardised magnitude distance for a measure looping through each control store 

# In[13]:


def calculate_magnitude_dist(measure, storeComparison, inputTable=preTrialMeasures):
    control_store_nbrs = inputTable[~inputTable["STORE_NBR"].isin([77, 86, 88])]["STORE_NBR"].unique()
    dists = pd.DataFrame()
    trial_store = inputTable[inputTable["STORE_NBR"] == storeComparison][measure]
    for control in control_store_nbrs:
        concat_df = abs(inputTable[inputTable["STORE_NBR"] == storeComparison].reset_index()[measure] - inputTable[inputTable["STORE_NBR"] == control].reset_index()[measure])
        concat_df["YEARMONTH"] = list(inputTable[inputTable["STORE_NBR"] == storeComparison]["YEARMONTH"])
        concat_df["Trial_Str"] = storeComparison
        concat_df["Ctrl_Str"] = control
        dists = pd.concat([dists, concat_df])
    for col in measure:
        dists[col] = 1 - ((dists[col] - dists[col].min()) / (dists[col].max() - dists[col].min()))
    dists["magnitude"] = dists[measure].mean(axis=1)
    return dists


# In[14]:


dist_table = pd.DataFrame()
for trial_num in [77, 86, 88]:
    dist_table = pd.concat([dist_table, calculate_magnitude_dist(["TOT_SALES", "nCustomers", "nTxnPerCust", "nChipsPerTxn", "avgPricePerUnit"], trial_num)])
    
dist_table.head(8)
dist_table


# In[ ]:





# #### Create a combined score composed of correlation and magnitude, by first merging the correlations table with the magnitude table.

#  We'll select control stores based on how similar monthly total sales in dollar amounts and monthly number of customers are to the trial stores by using correlation and magnitude distance.

# In[15]:


def combine_corr_dist(measure, storeComparison, inputTable=preTrialMeasures):
    corrs = calculate_correlation(measure, storeComparison, inputTable)
    dists = calculate_magnitude_dist(measure, storeComparison, inputTable)
    dists = dists.drop(measure, axis=1)
    combine = pd.merge(corrs, dists, on=["YEARMONTH", "Trial_Str", "Ctrl_Str"])
    return combine


# In[16]:


compare_metrics_table1 = pd.DataFrame()
for trial_num in [77, 86, 88]:
    compare_metrics_table1 = pd.concat([compare_metrics_table1, combine_corr_dist(["TOT_SALES"], trial_num)])


# In[17]:


corr_weight = 0.5
dist_weight = 1 - corr_weight


# In[ ]:





# #### Let's determine the top five highest composite score for each trial based on total sales

# In[18]:


grouped_comparison_table1 = compare_metrics_table1.groupby(["Trial_Str", "Ctrl_Str"], as_index=False).mean(numeric_only=True)
grouped_comparison_table1["CompScore"] = (corr_weight * grouped_comparison_table1["Corr_Score"]) + (dist_weight * grouped_comparison_table1["magnitude"])
for trial_num in compare_metrics_table1["Trial_Str"].unique():
    print(grouped_comparison_table1[grouped_comparison_table1["Trial_Str"] == trial_num].sort_values(by="CompScore", ascending=False).head(), '\n')


# Similarities based on total sales:
# 
# - Trial store 77: Stores 233, 188, 131
# - Trial store 86: Stores 155, 109, 138
# - Trial store 88: Stores 40, 4, 26

# In[19]:


compare_metrics_table2 = pd.DataFrame()
for trial_num in [77, 86, 88]:
    compare_metrics_table2 = pd.concat([compare_metrics_table2, combine_corr_dist(["nCustomers"], trial_num)])
     


# In[ ]:





# #### Let's determine the top five highest composite score for each trial based on no. of customers

# In[20]:


grouped_comparison_table2 = compare_metrics_table2.groupby(["Trial_Str", "Ctrl_Str"], as_index=False).mean(numeric_only=True)
grouped_comparison_table2["CompScore"] = (corr_weight * grouped_comparison_table2["Corr_Score"]) + (dist_weight * grouped_comparison_table2["magnitude"])
for trial_num in compare_metrics_table2["Trial_Str"].unique():
    print(grouped_comparison_table2[grouped_comparison_table2["Trial_Str"] == trial_num].sort_values(ascending=False, by="CompScore").head(), '\n')


# Similarities based on no. of customers:
# 
# - Trial store 77: Stores 233, 41, 17
# - Trial store 86: Stores 155, 225, 114
# - Trial store 88: Stores 237, 40, 199

# In[ ]:





# #### Let's determine the top composite score for each trial based on highest average total sales and no. of customers

# In[21]:


for trial_num in compare_metrics_table2["Trial_Str"].unique():
    a = grouped_comparison_table1[grouped_comparison_table1["Trial_Str"] == trial_num].sort_values(ascending=False, by="CompScore").set_index(["Trial_Str", "Ctrl_Str"])["CompScore"]
    b = grouped_comparison_table2[grouped_comparison_table2["Trial_Str"] == trial_num].sort_values(ascending=False, by="CompScore").set_index(["Trial_Str", "Ctrl_Str"])["CompScore"]
    print((pd.concat([a,b], axis=1).sum(axis=1)/2).sort_values(ascending=False).head(3), '\n')
     


# Final similarities based on highest average of both features combined:
# 
# - Trial store 77: Store 233
# - Trial store 86: Store 155
# - Trial store 88: Store 40

# In[ ]:





# Similarities based on total sales:
# 
# - Trial store 77: Stores 233, 188, 131
# - Trial store 86: Stores 155, 109, 138
# - Trial store 88: Stores 40, 4, 26
#     
# Similarities based on no. of customers:
# 
# - Trial store 77: Stores 233, 41, 17
# - Trial store 86: Stores 155, 225, 114
# - Trial store 88: Stores 237, 40, 199
#     
# Final similarities based on highest average of both features combined:
# 
# - Trial store 77: Store 233
# - Trial store 86: Store 155
# - Trial store 88: Store 40

# In[ ]:





# In[22]:


trial_control_dic = {77:233, 86:155, 88:40}
for key, val in trial_control_dic.items():
    preTrialMeasures[preTrialMeasures["STORE_NBR"].isin([key, val])].groupby(
        ["YEARMONTH", "STORE_NBR"]).sum()["TOT_SALES"].unstack().plot.bar()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Trial Store "+str(key)+" and Control Store "+str(val)+" - TOT_SALES")
    plt.show()
    preTrialMeasures[preTrialMeasures["STORE_NBR"].isin([key, val])].groupby(
    ["YEARMONTH", "STORE_NBR"]).sum()["nCustomers"].unstack().plot.bar()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Trial Store "+str(key)+" and Control Store "+str(val)+" - nCustomer")
    plt.show()


# In[ ]:





# Next we'll compare the performance of Trial stores to Control stores during the trial period. To ensure their performance is comparable during Trial period, we need to scale (multiply to ratio of trial / control) all of Control stores' performance to Trial store's performance during pre-trial. Starting with TOT_SALES.

# In[23]:


#Ratio of Store 77 and its Control store.
sales_ratio_77 = preTrialMeasures[preTrialMeasures["STORE_NBR"] == 77]["TOT_SALES"].sum() / preTrialMeasures[preTrialMeasures["STORE_NBR"] == 233]["TOT_SALES"].sum()

#Ratio of Store 86 and its Control store.
sales_ratio_86 = preTrialMeasures[preTrialMeasures["STORE_NBR"] == 86]["TOT_SALES"].sum() / preTrialMeasures[preTrialMeasures["STORE_NBR"] == 155]["TOT_SALES"].sum()

#Ratio of Store 77 and its Control store.
sales_ratio_88 = preTrialMeasures[preTrialMeasures["STORE_NBR"] == 88]["TOT_SALES"].sum() / preTrialMeasures[preTrialMeasures["STORE_NBR"] == 40]["TOT_SALES"].sum()
     


# In[24]:


trial_full_observ = full_observ[(full_observ["YEARMONTH"] >= 201902) & (full_observ["YEARMONTH"] <= 201904)]
scaled_sales_control_stores = full_observ[full_observ["STORE_NBR"].isin([233, 155, 40])][["STORE_NBR", "YEARMONTH", "TOT_SALES"]]

def scaler(row):
    if row["STORE_NBR"] == 233:
        return row["TOT_SALES"] * sales_ratio_77
    elif row["STORE_NBR"] == 155:
        return row["TOT_SALES"] * sales_ratio_86
    elif row["STORE_NBR"] == 40:
        return row["TOT_SALES"] * sales_ratio_88

scaled_sales_control_stores["ScaledSales"] = scaled_sales_control_stores.apply(lambda row: scaler(row), axis=1)

trial_scaled_sales_control_stores = scaled_sales_control_stores[(scaled_sales_control_stores["YEARMONTH"] >= 201902) & (scaled_sales_control_stores["YEARMONTH"] <= 201904)]
pretrial_scaled_sales_control_stores = scaled_sales_control_stores[scaled_sales_control_stores["YEARMONTH"] < 201902]

percentage_diff = {}

for trial, control in trial_control_dic.items():
    a = trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == control]
    b = trial_full_observ[trial_full_observ["STORE_NBR"] == trial][["STORE_NBR", "YEARMONTH", "TOT_SALES"]]
    percentage_diff[trial] = b["TOT_SALES"].sum() / a["ScaledSales"].sum()
    b[["YEARMONTH", "TOT_SALES"]].merge(a[["YEARMONTH", "ScaledSales"]],on="YEARMONTH").set_index("YEARMONTH").rename(columns={"ScaledSales":"Scaled_Control_Sales", "TOT_SALES":"Trial_Sales"}).plot.bar()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Trial Store "+str(trial)+" and Control Store "+str(control))


# In[25]:


percentage_diff


# In[26]:


temp1 = scaled_sales_control_stores.sort_values(by=["STORE_NBR", "YEARMONTH"], ascending=[False, True]).reset_index().drop(["TOT_SALES", "index"], axis=1)
temp2 = full_observ[full_observ["STORE_NBR"].isin([77,86,88])][["STORE_NBR", "YEARMONTH", "TOT_SALES"]].reset_index().drop(["index", "YEARMONTH"], axis=1)
scaledsales_vs_trial = pd.concat([temp1, temp2], axis=1)
scaledsales_vs_trial.columns = ["c_STORE_NBR", "YEARMONTH", "c_ScaledSales", "t_STORE_NBR", "t_TOT_SALES"]
scaledsales_vs_trial["Sales_Percentage_Diff"] = (scaledsales_vs_trial["t_TOT_SALES"] - scaledsales_vs_trial["c_ScaledSales"]) / (((scaledsales_vs_trial["t_TOT_SALES"] + scaledsales_vs_trial["c_ScaledSales"])/2))
def label_period(cell):
    if cell < 201902:
        return "pre"
    elif cell > 201904:
        return "post"
    else:
        return "trial"
scaledsales_vs_trial["trial_period"] = scaledsales_vs_trial["YEARMONTH"].apply(lambda cell: label_period(cell))
scaledsales_vs_trial[scaledsales_vs_trial["trial_period"] == "trial"]


# Check significance of Trial minus Control stores TOT_SALES Percentage Difference Pre-Trial vs Trial.
# 
# Step 1: Check null hypothesis of 0 difference between control store's Pre-Trial and Trial period performance.
# 
# Step 2: Proof control and trial stores are similar statistically
# 
# Check p-value of control store's Pre-Trial vs Trial store's Pre-Trial. If <5%, it is significantly different. If >5%, it is not significantly different (similar).
# 
# Step 3: After checking Null Hypothesis of first 2 step to be true, we can check Null Hypothesis of percentage difference between Trial and Control stores during pre-trial is the same as during trial.
# 
# Check T-Value of Percentage Difference of each Trial month (Feb, March, April 2019). Mean is mean of Percentage Difference during pre-trial. Standard deviation is stdev of Percentage Difference during pre-trial. Formula is Trial month's Percentage Difference minus Mean, divided by Standard deviation. Compare each T-Value with 95% percentage significance critical t-value of 6 degrees of freedom (7 months of sample - 1)

# #### Step 1: Check null hypothesis of 0 difference between control store's Pre-Trial and Trial period performance.

# In[27]:


from scipy.stats import ttest_ind, t

# Step 1
for num in [40, 155, 233]:
    print("Store", num)
    print(ttest_ind(pretrial_scaled_sales_control_stores[pretrial_scaled_sales_control_stores["STORE_NBR"] == num]["ScaledSales"],
                   trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == num]["ScaledSales"],
                   equal_var=False), '\n')
    #print(len(pretrial_scaled_sales_control_stores[pretrial_scaled_sales_control_stores["STORE_NBR"] == num]["ScaledSales"]), len(trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == num]["ScaledSales"]))
    
alpha = 0.05
print("Critical t-value for 95% confidence interval:")
print(t.ppf((alpha/2, 1-alpha/2), df=min([len(pretrial_scaled_sales_control_stores[pretrial_scaled_sales_control_stores["STORE_NBR"] == num]),
                       len(trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == num])])-1))


# In[28]:


a = pretrial_scaled_sales_control_stores[pretrial_scaled_sales_control_stores["STORE_NBR"] == 40]["ScaledSales"]
b = trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == 40]["ScaledSales"]
     


# Null hypothesis is true. There is not any statistically significant difference between control store's scaled Pre-Trial and Trial period sales.

# #### Step 2: Proof control and trial stores are similar statistically

# In[29]:


# Step 2
for trial, cont in trial_control_dic.items():
    print("Trial store:", trial, ", Control store:", cont)
    print(ttest_ind(preTrialMeasures[preTrialMeasures["STORE_NBR"] == trial]["TOT_SALES"],
                   pretrial_scaled_sales_control_stores[pretrial_scaled_sales_control_stores["STORE_NBR"] == cont]["ScaledSales"],
                   equal_var=True), '\n')
    #print(len(pretrial_full_observ[pretrial_full_observ["STORE_NBR"] == trial]["TOT_SALES"]),len(pretrial_scaled_sales_control_stores[pretrial_scaled_sales_control_stores["STORE_NBR"] == cont]["ScaledSales"]))

alpha = 0.05
print("Critical t-value for 95% confidence interval:")
print(t.ppf((alpha/2, 1-alpha/2), df=len(preTrialMeasures[preTrialMeasures["STORE_NBR"] == trial])-1))


# The null hypothesis holds true, indicating that there is no statistically significant distinction between the sales performance of the Trial store and the scaled sales performance of the Control store during the pre-trial period.

# #### Step 3: Check Null Hypothesis of percentage difference between Trial and Control stores during pre-trial is the same as during trial.

# In[30]:


# Step 3
for trial, cont in trial_control_dic.items():
    print("Trial store:", trial, ", Control store:", cont)
    temp_pre = scaledsales_vs_trial[(scaledsales_vs_trial["c_STORE_NBR"] == cont) & (scaledsales_vs_trial["trial_period"]=="pre")]
    std = temp_pre["Sales_Percentage_Diff"].std()
    mean = temp_pre["Sales_Percentage_Diff"].mean()
    #print(std, mean)
    for t_month in scaledsales_vs_trial[scaledsales_vs_trial["trial_period"] == "trial"]["YEARMONTH"].unique():
        pdif = scaledsales_vs_trial[(scaledsales_vs_trial["YEARMONTH"] == t_month) & (scaledsales_vs_trial["t_STORE_NBR"] == trial)]["Sales_Percentage_Diff"]
        print(t_month,":",(float(pdif)-mean)/std)
    print('\n')
    
print("Critical t-value for 95% confidence interval:")
conf_intv_95 = t.ppf(0.95, df=len(temp_pre)-1)
print(conf_intv_95)
     


# There are 3 months' increase in performance that are statistically significant (Above the 95% confidence interval t-score):
# 
# March and April trial months for trial store 77
# 
# March trial months for trial store 86

# In[ ]:





# In[31]:


for trial, control in trial_control_dic.items():
    a = trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == control].rename(columns={"TOT_SALES": "control_TOT_SALES"})
    b = trial_full_observ[trial_full_observ["STORE_NBR"] == trial][["STORE_NBR", "YEARMONTH", "TOT_SALES"]].rename(columns={"TOT_SALES": "trial_TOT_SALES"})
    comb = b[["YEARMONTH", "trial_TOT_SALES"]].merge(a[["YEARMONTH", "control_TOT_SALES"]],on="YEARMONTH").set_index("YEARMONTH")
    comb.plot.bar()
    cont_sc_sales = trial_scaled_sales_control_stores[trial_scaled_sales_control_stores["STORE_NBR"] == control]["TOT_SALES"]
    std = scaledsales_vs_trial[(scaledsales_vs_trial["c_STORE_NBR"] == control) & (scaledsales_vs_trial["trial_period"]=="pre")]["Sales_Percentage_Diff"].std()
    thresh95 = cont_sc_sales.mean() + (cont_sc_sales.mean() * std * 2)
    thresh5 = cont_sc_sales.mean() - (cont_sc_sales.mean() * std * 2)
    plt.axhline(y=thresh95,linewidth=1, color='b', label="95% threshold")
    plt.axhline(y=thresh5,linewidth=1, color='r', label="5% threshold")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Trial Store "+str(trial)+" and Control Store "+str(control)+" - Total Sales")
    plt.savefig("TS {} and CS {} - TOT_SALES.png".format(trial,control), bbox_inches="tight")


# In[32]:


#Ratio of Store 77 and its Control store.
ncust_ratio_77 = preTrialMeasures[preTrialMeasures["STORE_NBR"] == 77]["nCustomers"].sum() / preTrialMeasures[preTrialMeasures["STORE_NBR"] == 233]["nCustomers"].sum()

#Ratio of Store 86 and its Control store.
ncust_ratio_86 = preTrialMeasures[preTrialMeasures["STORE_NBR"] == 86]["nCustomers"].sum() / preTrialMeasures[preTrialMeasures["STORE_NBR"] == 155]["nCustomers"].sum()

#Ratio of Store 77 and its Control store.
ncust_ratio_88 = preTrialMeasures[preTrialMeasures["STORE_NBR"] == 88]["nCustomers"].sum() / preTrialMeasures[preTrialMeasures["STORE_NBR"] == 40]["nCustomers"].sum()
     


# In[33]:


#trial_full_observ = full_observ[(full_observ["YEARMONTH"] >= 201902) & (full_observ["YEARMONTH"] <= 201904)]
scaled_ncust_control_stores = full_observ[full_observ["STORE_NBR"].isin([233, 155, 40])][["STORE_NBR", "YEARMONTH", "nCustomers"]]

def scaler_c(row):
    if row["STORE_NBR"] == 233:
        return row["nCustomers"] * ncust_ratio_77
    elif row["STORE_NBR"] == 155:
        return row["nCustomers"] * ncust_ratio_86
    elif row["STORE_NBR"] == 40:
        return row["nCustomers"] * ncust_ratio_88

scaled_ncust_control_stores["ScaledNcust"] = scaled_ncust_control_stores.apply(lambda row: scaler_c(row), axis=1)

trial_scaled_ncust_control_stores = scaled_ncust_control_stores[(scaled_ncust_control_stores["YEARMONTH"] >= 201902) & (scaled_ncust_control_stores["YEARMONTH"] <= 201904)]
pretrial_scaled_ncust_control_stores = scaled_ncust_control_stores[scaled_ncust_control_stores["YEARMONTH"] < 201902]

ncust_percentage_diff = {}

for trial, control in trial_control_dic.items():
    a = trial_scaled_ncust_control_stores[trial_scaled_ncust_control_stores["STORE_NBR"] == control]
    b = trial_full_observ[trial_full_observ["STORE_NBR"] == trial][["STORE_NBR", "YEARMONTH", "nCustomers"]]
    ncust_percentage_diff[trial] = b["nCustomers"].sum() / a["ScaledNcust"].sum()
    b[["YEARMONTH", "nCustomers"]].merge(a[["YEARMONTH", "ScaledNcust"]],on="YEARMONTH").set_index("YEARMONTH").rename(columns={"ScaledSales":"Scaled_Control_nCust", "TOT_SALES":"Trial_nCust"}).plot.bar()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Trial Store "+str(trial)+" and Control Store "+str(control))


# In[34]:


ncust_percentage_diff


# In[35]:


temp1 = scaled_ncust_control_stores.sort_values(by=["STORE_NBR", "YEARMONTH"], ascending=[False, True]).reset_index().drop(["nCustomers", "index"], axis=1)
temp2 = full_observ[full_observ["STORE_NBR"].isin([77,86,88])][["STORE_NBR", "YEARMONTH", "nCustomers"]].reset_index().drop(["index", "YEARMONTH"], axis=1)
scaledncust_vs_trial = pd.concat([temp1, temp2], axis=1)
scaledncust_vs_trial.columns = ["c_STORE_NBR", "YEARMONTH", "c_ScaledNcust", "t_STORE_NBR", "t_nCustomers"]
scaledncust_vs_trial["nCust_Percentage_Diff"] = (scaledncust_vs_trial["t_nCustomers"] - scaledncust_vs_trial["c_ScaledNcust"]) / (((scaledncust_vs_trial["t_nCustomers"] + scaledncust_vs_trial["c_ScaledNcust"])/2))

scaledncust_vs_trial["trial_period"] = scaledncust_vs_trial["YEARMONTH"].apply(lambda cell: label_period(cell))
scaledncust_vs_trial[scaledncust_vs_trial["trial_period"] == "trial"]


# In[ ]:





# Check significance of Trial minus Control stores nCustomers Percentage Difference Pre-Trial vs Trial.
# 
# Step 1: Check null hypothesis of 0 difference between control store's Pre-Trial and Trial period performance.
# 
# Step 2: Proof control and trial stores are similar statistically
# 
# Step 3: After checking Null Hypothesis of first 2 step to be true, we can check Null Hypothesis of Percentage Difference between Trial and Control stores during pre-trial is the same as during trial.

# #### Step 1: Check null hypothesis of 0 difference between control store's Pre-Trial and Trial period performance.

# In[36]:


for num in [40, 155, 233]:
    print("Store", num)
    print(ttest_ind(pretrial_scaled_ncust_control_stores[pretrial_scaled_ncust_control_stores["STORE_NBR"] == num]["ScaledNcust"],
                   trial_scaled_ncust_control_stores[trial_scaled_ncust_control_stores["STORE_NBR"] == num]["ScaledNcust"],
                   equal_var=False), '\n')
    
alpha = 0.05
print("Critical t-value for 95% confidence interval:")
print(t.ppf((alpha/2, 1-alpha/2), df=min([len(pretrial_scaled_ncust_control_stores[pretrial_scaled_ncust_control_stores["STORE_NBR"] == num]),
                       len(trial_scaled_ncust_control_stores[trial_scaled_ncust_control_stores["STORE_NBR"] == num])])-1))
     


# In[ ]:





# #### Step 2: Proof control and trial stores are similar statistically

# In[37]:


for trial, cont in trial_control_dic.items():
    print("Trial store:", trial, ", Control store:", cont)
    print(ttest_ind(preTrialMeasures[preTrialMeasures["STORE_NBR"] == trial]["nCustomers"],
                   pretrial_scaled_ncust_control_stores[pretrial_scaled_ncust_control_stores["STORE_NBR"] == cont]["ScaledNcust"],
                   equal_var=True), '\n')

alpha = 0.05
print("Critical t-value for 95% confidence interval:")
print(t.ppf((alpha/2, 1-alpha/2), df=len(preTrialMeasures[preTrialMeasures["STORE_NBR"] == trial])-1))
     


# In[ ]:





# #### Step 3: Check Null Hypothesis of Percentage Difference between Trial and Control stores during pre-trial is the same as during trial.

# In[38]:


for trial, cont in trial_control_dic.items():
    print("Trial store:", trial, ", Control store:", cont)
    temp_pre = scaledncust_vs_trial[(scaledncust_vs_trial["c_STORE_NBR"] == cont) & (scaledncust_vs_trial["trial_period"]=="pre")]
    std = temp_pre["nCust_Percentage_Diff"].std()
    mean = temp_pre["nCust_Percentage_Diff"].mean()
    #print(std, mean)
    for t_month in scaledncust_vs_trial[scaledncust_vs_trial["trial_period"] == "trial"]["YEARMONTH"].unique():
        pdif = scaledncust_vs_trial[(scaledncust_vs_trial["YEARMONTH"] == t_month) & (scaledncust_vs_trial["t_STORE_NBR"] == trial)]["nCust_Percentage_Diff"]
        print(t_month,":",(float(pdif)-mean)/std)
    print('\n')
    
print("Critical t-value for 95% confidence interval:")
conf_intv_95 = t.ppf(0.95, df=len(temp_pre)-1)
print(conf_intv_95)


# There are 5 months' increase in performance that are statistically significant (Above the 95% confidence interval t-score):
# 
# March and April trial months for trial store 77
# 
# Feb, March and April trial months for trial store 86

# In[39]:


for trial, control in trial_control_dic.items():
    a = trial_scaled_ncust_control_stores[trial_scaled_ncust_control_stores["STORE_NBR"] == control].rename(columns={"nCustomers": "control_nCustomers"})
    b = trial_full_observ[trial_full_observ["STORE_NBR"] == trial][["STORE_NBR", "YEARMONTH", "nCustomers"]].rename(columns={"nCustomers": "trial_nCustomers"})
    comb = b[["YEARMONTH", "trial_nCustomers"]].merge(a[["YEARMONTH", "control_nCustomers"]],on="YEARMONTH").set_index("YEARMONTH")
    comb.plot.bar()
    cont_sc_ncust = trial_scaled_ncust_control_stores[trial_scaled_ncust_control_stores["STORE_NBR"] == control]["nCustomers"]
    std = scaledncust_vs_trial[(scaledncust_vs_trial["c_STORE_NBR"] == control) & (scaledncust_vs_trial["trial_period"]=="pre")]["nCust_Percentage_Diff"].std()
    thresh95 = cont_sc_ncust.mean() + (cont_sc_ncust.mean() * std * 2)
    thresh5 = cont_sc_ncust.mean() - (cont_sc_ncust.mean() * std * 2)
    plt.axhline(y=thresh95,linewidth=1, color='b', label="95% threshold")
    plt.axhline(y=thresh5,linewidth=1, color='r', label="5% threshold")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Trial Store "+str(trial)+" and Control Store "+str(control)+" - Number of Customers")
    plt.savefig("TS {} and CS {} - nCustomers.png".format(trial,control), bbox_inches="tight")


# In[ ]:





# We can see that Trial store 77 sales for Feb, March, and April exceeds 95% threshold of control store. Same goes to store 86 sales for all 3 trial months.
# 
# 1. Trial store 77: Control store 233
# 2. Trial store 86: Control store 155
# 3. Trial store 88: Control store 40
# 
# The analysis reveals that Trial stores 77 and 86 exhibited a significant increase in total sales and number of customers during the three-month trial period, exceeding the 95% threshold of their respective Control stores. Conversely, Trial store 88 did not exhibit a significant increase in performance during the trial period compared to its Control store. It is possible that there were certain factors that distinguished Trial store 88 from the other two Trial stores, which could have contributed to this difference in performance. However, on the whole, the trial showed a positive and statistically significant outcome.

# In[ ]:





# In[ ]:




