import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from google.colab import drive

# Load sales data
sales_train_df = pd.read_csv('./train.csv')
print(sales_train_df.head(5))

# Display summary statistics and information
print(sales_train_df.info())
print(sales_train_df.describe())

# Load store information data
store_info_df = pd.read_csv('./store.csv')
print(store_info_df.head(5))

# Display summary statistics and information for store data
print(store_info_df.info())
print(store_info_df.describe())

# Check for missing values in sales_train_df
sns.heatmap(sales_train_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")

# Visualize distributions
sales_train_df.hist(bins=30, figsize=(20, 20), color='r')
print(f"Max customers: {sales_train_df['Customers'].max()}")

# Separate open and closed stores
closed_train_df = sales_train_df[sales_train_df['Open'] == 0]
open_train_df = sales_train_df[sales_train_df['Open'] == 1]

# Display counts of open and closed stores
print(f"Total = {len(sales_train_df)}")
print(f"Number of closed stores = {len(closed_train_df)}")
print(f"Number of open stores = {len(open_train_df)}")

# Remove closed stores
sales_train_df = sales_train_df[sales_train_df['Open'] == 1]

# Drop the 'Open' column as it is no longer needed
sales_train_df.drop(columns=['Open'], inplace=True)
print(sales_train_df.describe())

# Check for missing values in store_info_df
sns.heatmap(store_info_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")

# Handle missing values in store_info_df
store_info_df['CompetitionDistance'].fillna(store_info_df['CompetitionDistance'].mean(), inplace=True)
cols_to_fill = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CompetitionOpenSinceYear',
                'CompetitionOpenSinceMonth']
store_info_df[cols_to_fill] = store_info_df[cols_to_fill].fillna(0)
sns.heatmap(store_info_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")

# Visualize distributions in store_info_df
store_info_df.hist(bins=30, figsize=(20, 20), color='r')

# Merge sales and store dataframes
sales_train_all_df = pd.merge(sales_train_df, store_info_df, how='inner', on='Store')
sales_train_all_df.to_csv('/content/drive/My Drive/sales_train_all.csv', index=False)

# Correlation matrix
correlations = sales_train_all_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlations, annot=True)

# Add year, month, and day columns
sales_train_all_df['Year'] = pd.DatetimeIndex(sales_train_all_df['Date']).year
sales_train_all_df['Month'] = pd.DatetimeIndex(sales_train_all_df['Date']).month
sales_train_all_df['Day'] = pd.DatetimeIndex(sales_train_all_df['Date']).day

# Average sales and customers per month
plt.figure()
sales_train_all_df.groupby('Month')['Sales'].mean().plot(figsize=(10, 5), marker='o', color='r',
                                                         title='Average Sales Per Month')
plt.figure()
sales_train_all_df.groupby('Month')['Customers'].mean().plot(figsize=(10, 5), marker='^', color='b',
                                                             title='Average Customers Per Month')

# Average sales and customers per day
plt.figure()
sales_train_all_df.groupby('Day')['Sales'].mean().plot(figsize=(10, 5), marker='o', color='r',
                                                       title='Average Sales Per Day')
plt.figure()
sales_train_all_df.groupby('Day')['Customers'].mean().plot(figsize=(10, 5), marker='^', color='b',
                                                           title='Average Customers Per Day')

# Average sales and customers per day of the week
plt.figure()
sales_train_all_df.groupby('DayOfWeek')['Sales'].mean().plot(figsize=(10, 5), marker='o', color='r',
                                                             title='Average Sales Per Day of the Week')
plt.figure()
sales_train_all_df.groupby('DayOfWeek')['Customers'].mean().plot(figsize=(10, 5), marker='^', color='b',
                                                                 title='Average Customers Per Day of the Week')

# Sales by store type over time
fig, ax = plt.subplots(figsize=(20, 10))
sales_train_all_df.groupby(['Date', 'StoreType']).mean()['Sales'].unstack().plot(ax=ax)

# Sales and customers based on promo
plt.figure(figsize=[15, 10])
plt.subplot(211)
sns.barplot(x='Promo', y='Sales', data=sales_train_all_df)
plt.subplot(212)
sns.barplot(x='Promo', y='Customers', data=sales_train_all_df)

# Violin plots for promo
plt.figure(figsize=[15, 10])
plt.subplot(211)
sns.violinplot(x='Promo', y='Sales', data=sales_train_all_df)
plt.subplot(212)
sns.violinplot(x='Promo', y='Customers', data=sales_train_all_df)


def sales_prediction(store_id, sales_df, periods):
    """Function to predict sales for a specific store using Prophet."""
    store_sales_df = sales_df[sales_df['Store'] == store_id][['Date', 'Sales']]
    store_sales_df = store_sales_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    store_sales_df = store_sales_df.sort_values('ds')

    model = Prophet()
    model.fit(store_sales_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    return forecast


sales_prediction(10, sales_train_all_df, 60)


def sales_prediction_with_holidays(store_id, sales_df, holidays, periods):
    """Function to predict sales for a specific store using Prophet with holidays."""
    store_sales_df = sales_df[sales_df['Store'] == store_id][['Date', 'Sales']]
    store_sales_df = store_sales_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    store_sales_df = store_sales_df.sort_values('ds')

    model = Prophet(holidays=holidays)
    model.fit(store_sales_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    return forecast


# Extract holidays
school_holidays = sales_train_all_df[sales_train_all_df['SchoolHoliday'] == 1]['Date'].values
state_holidays = sales_train_all_df[sales_train_all_df['StateHoliday'].isin(['a', 'b', 'c'])]['Date'].values

# Create holidays dataframe
state_holidays_df = pd.DataFrame({'ds': pd.to_datetime(state_holidays), 'holiday': 'state_holiday'})
school_holidays_df = pd.DataFrame({'ds': pd.to_datetime(school_holidays), 'holiday': 'school_holiday'})

# Combine holidays
holidays_df = pd.concat([state_holidays_df, school_holidays_df])
print(holidays_df)

# Predict sales for a store with holidays
sales_prediction_with_holidays(6, sales_train_all_df, holidays_df, 60)
