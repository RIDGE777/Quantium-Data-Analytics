# QUANTIUM - DATA - ANALYTICS

## TASK 1 – RETAIL STRATEGY AND ANALYTICS

This task consists of 2 datasets, the transaction data and customer purchase behaviors data. I was tasked with cleaning and exploring the 2 datasets before aggregating the 2 datasets and analyzing the data to check for trends and insights from the customer purchase behaviors.

### Data Cleaning

1) **Transaction Data Dataset**

The following are the data-cleaning steps I took to ensure the data was accurate, consistent, and ready for analysis:

* Changed the datatype of the DATE column from int64 to DateTime.
* Deleted salsa products from the data which are not chips by checking the most common words and how many times the words appear in the PROD_NAME column.
* Created a new column for chip package sizes, PACK_SIZE_GRAMS, by extracting the pack size data from the PROD_NAME column.
* Created a new column for the chip brands, BRAND, by extracting the brand names from the PROD_NAME column. I then cleaned the BRAND column by checking for spelling mistakes in the brand names and correcting them.
* Checked for outliers in the data by looking at the data summary statistics. I deleted the data that was a possible outlier in the dataset.

2
