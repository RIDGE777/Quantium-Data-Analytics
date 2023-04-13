# QUANTIUM - DATA - ANALYTICS

## TASK 1 – RETAIL STRATEGY AND ANALYTICS

This task consists of 2 datasets, the transaction data and customer purchase behaviors data. I was tasked with cleaning and exploring the 2 datasets before aggregating the 2 datasets and analyzing the data to check for trends and insights from the customer purchase behaviors.

## Data Cleaning

1. **Transaction Data Dataset**

The following are the data-cleaning steps I took to ensure the data was accurate, consistent, and ready for analysis:

* Changed the datatype of the DATE column from int64 to DateTime.
* Deleted salsa products from the data which are not chips by checking the most common words and how many times the words appear in the PROD_NAME column.
* Created a new column for chip package sizes, PACK_SIZE_GRAMS, by extracting the pack size data from the PROD_NAME column.
* Created a new column for the chip brands, BRAND, by extracting the brand names from the PROD_NAME column. I then cleaned the BRAND column by checking for spelling mistakes in the brand names and correcting them.
* Checked for outliers in the data by looking at the data summary statistics. I deleted the data that was a possible outlier in the dataset.

2. **Customer Data Dataset**

The following are the data-cleaning steps I took to ensure the data was accurate, consistent, and ready for analysis:
* Checked and ensured that there was no nulls in the dataset that could affect our analysis

I then merged the 2 datasets and exported the resulting file into an Excel worksheet, customer_transaction_data, ready for analysis and visualization. 

## Data Analysis

The following are the guiding questions I had to identify trends and insights in the data:

1. How many customers are in each segment? 
2. Who spends the most on chips (total sales)? 
3. What is the relationship between total sales and customer segments and the number of customers per customer segment?
4. How many chips are bought per customer by segment?
5. What is the relationship between total chip purchases vs all customer segments combined?
6. What's the average chip price by customer segment?

I then did a deep dive into the mainstream- young singles/couples customer segment to understand their purchase behaviours to generate insights and give recommendations on how to do more marketing targeting this segment to maximize sales in this category.



## TASK 2 - EXPERIMENTATION AND UPLIFT TESTING

I was tasked to evaluate the performance of a store trial which was performed in stores 77, 86, and 88 using the combined transaction and purchase behavior datasets.

I had to break down the monthly data for each store by total sales, the total number of customers, and the average number of transactions per customer.

I then created a measure to compare different control stores to each of the trial stores.

Once I selected my control stores, I compared each trial and control pair during the trial period. I tested if total sales are significantly different in the trial period and checked if the driver of change is more purchasing customers or more purchases per customer.

The following are the actions I took in task 2:

1. Added a new month ID, YEARMONTH, column in the data with the format yyyymm.
2. Created a new data frame measureOverTime to calculate total sales, number of customers, transactions per customer, chips per customer, and the average price per unit for each store and month
3. Filtered to the pre-trial period and stores with full observation periods (12 months) - control stores.
4. Created a function to calculate correlation for a measure, looping through each control store.
5. Created a function to calculate a standardized magnitude distance for a measure looping through each control store.
6. Created a combined score composed of correlation and magnitude by merging the correlations table with the magnitude table.
7. Determined the top five highest composite scores for each trial based on total sales, number of customers, and the highest average of total sales and number of customers.

I compared the performance of Trial stores to Control stores during the trial period. To ensure their performance is comparable during the Trial period, I scaled (multiply to the ratio of trial/control) all of the Control stores' performance to the Trial store's performance during pre-trial starting with TOT_SALES.

Checked significance of Trial Minus Control stores TOT_SALES Percentage Difference Pre-Trial Vs Trial.

Step 1: Checked the null hypothesis of 0 difference between the control store's Pre-Trial and Trial period performance.

Step 2: Proofed control and trial stores are similar statistically

Checked the p-value of the control store's Pre-Trial vs the Trial Store’s Pre-Trial. If <5%, it is significantly different. If >5%, it is not significantly different (similar).

Step 3: After checking the Null Hypothesis of the first 2 steps to be true, I checked the Null Hypothesis of the percentage difference between Trial and Control stores during pre-trial is the same as during the trial.

Checked the T-Value of the Percentage Difference of each Trial month (Feb, March, and April 2019). The mean is the mean of Percentage Difference during pre-trial. Standard deviation is the stdev of Percentage Difference during pre-trial. Formula is Trial Month’s Percentage Difference Minus Mean, divided by Standard deviation. Compare each T-Value with 95% percentage significance critical t-value of 6 degrees of freedom (7 months of sample - 1)

The analysis revealed that Trial stores 77 and 86 exhibited a significant increase in total sales and number of customers during the three-month trial period, exceeding the 95% threshold of their respective Control stores. Conversely, Trial store 88 did not exhibit a significant increase in performance during the trial period compared to its Control store. It is possible that there were certain factors that distinguished Trial store 88 from the other two Trial stores, which could have contributed to this difference in performance. However, on the whole, the trial showed a positive and statistically significant outcome.


## TASK 3 - PROJECT PRESENTATION

This was the last task of the project whereby I created a presentation of my analysis with visuals that would be easily understandable to stakeholders with recommendations on how to improve sales and if the experiment should be rolled out in all the other stores.
