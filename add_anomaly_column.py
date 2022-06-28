"""
Dividing the data into 48 categories.
24 hours on weekdays and 24 hours during the weekend.
Per category the top 0.3% is made an anomaly.
A column Anomaly is added to the data (csv file) in which true means anomaly and False means normal data point.
The labelled data is then stored in a new csv file.
Afterwards this labelled data is checked and, if needed, altered manually (not part of the code).
"""

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns

#read data from csv file
df = pd.read_csv(r'K:\USER\MaartenR\WorkWork\2018 - Anomaly Detection JIPS\Dataset\jips_dmch_uur_nogaps.csv')

# make list per hour for weekdays
hour_average00 = []
hour_average01 = []
hour_average02 = []
hour_average03 = []
hour_average04 = []
hour_average05 = []
hour_average06 = []
hour_average07 = []
hour_average08 = []
hour_average09 = []
hour_average10 = []
hour_average11 = []
hour_average12 = []
hour_average13 = []
hour_average14 = []
hour_average15 = []
hour_average16 = []
hour_average17 = []
hour_average18 = []
hour_average19 = []
hour_average20 = []
hour_average21 = []
hour_average22 = []
hour_average23 = []

# make list per hour for weekends
weekend_hour_average00 = []
weekend_hour_average01 = []
weekend_hour_average02 = []
weekend_hour_average03 = []
weekend_hour_average04 = []
weekend_hour_average05 = []
weekend_hour_average06 = []
weekend_hour_average07 = []
weekend_hour_average08 = []
weekend_hour_average09 = []
weekend_hour_average10 = []
weekend_hour_average11 = []
weekend_hour_average12 = []
weekend_hour_average13 = []
weekend_hour_average14 = []
weekend_hour_average15 = []
weekend_hour_average16 = []
weekend_hour_average17 = []
weekend_hour_average18 = []
weekend_hour_average19 = []
weekend_hour_average20 = []
weekend_hour_average21 = []
weekend_hour_average22 = []
weekend_hour_average23 = []

# first day is monday 15 October 2015
day = 1
i = 0
# for all data points in csv file
while i< len(df):
    # if weekend add the hours to their respective weekend hour list
    if day % 6 == 0 or day % 7 == 0:
        # try is used because in the last round not all lists can be filled
        # since the data doesn't start and end at the same time
        try:
            weekend_hour_average00.append(df.DateTime[i])
            weekend_hour_average01.append(df.DateTime[i+1])
            weekend_hour_average02.append(df.DateTime[i+2])
            weekend_hour_average03.append(df.DateTime[i+3])
            weekend_hour_average04.append(df.DateTime[i+4])
            weekend_hour_average05.append(df.DateTime[i+5])
            weekend_hour_average06.append(df.DateTime[i+6])
            weekend_hour_average07.append(df.DateTime[i+7])
            weekend_hour_average08.append(df.DateTime[i+8])
            weekend_hour_average09.append(df.DateTime[i+9])
            weekend_hour_average10.append(df.DateTime[i+10])
            weekend_hour_average11.append(df.DateTime[i+11])
            weekend_hour_average12.append(df.DateTime[i+12])
            weekend_hour_average13.append(df.DateTime[i+13])
            weekend_hour_average14.append(df.DateTime[i+14])
            weekend_hour_average15.append(df.DateTime[i+15])
            weekend_hour_average16.append(df.DateTime[i+16])
            weekend_hour_average17.append(df.DateTime[i+17])
            weekend_hour_average18.append(df.DateTime[i+18])
            weekend_hour_average19.append(df.DateTime[i+19])
            weekend_hour_average20.append(df.DateTime[i+20])
            weekend_hour_average21.append(df.DateTime[i+21])
            weekend_hour_average22.append(df.DateTime[i+22])
            weekend_hour_average23.append(df.DateTime[i+23])
        except:
            print("End of file")
    else:
        try:
            # try is used because in the last round not all lists can be filled
            # since the data doesn't start and end at the same time
            hour_average00.append(df.DateTime[i])
            hour_average01.append(df.DateTime[i+1])
            hour_average02.append(df.DateTime[i+2])
            hour_average03.append(df.DateTime[i+3])
            hour_average04.append(df.DateTime[i+4])
            hour_average05.append(df.DateTime[i+5])
            hour_average06.append(df.DateTime[i+6])
            hour_average07.append(df.DateTime[i+7])
            hour_average08.append(df.DateTime[i+8])
            hour_average09.append(df.DateTime[i+9])
            hour_average10.append(df.DateTime[i+10])
            hour_average11.append(df.DateTime[i+11])
            hour_average12.append(df.DateTime[i+12])
            hour_average13.append(df.DateTime[i+13])
            hour_average14.append(df.DateTime[i+14])
            hour_average15.append(df.DateTime[i+15])
            hour_average16.append(df.DateTime[i+16])
            hour_average17.append(df.DateTime[i+17])
            hour_average18.append(df.DateTime[i+18])
            hour_average19.append(df.DateTime[i+19])
            hour_average20.append(df.DateTime[i+20])
            hour_average21.append(df.DateTime[i+21])
            hour_average22.append(df.DateTime[i+22])
            hour_average23.append(df.DateTime[i+23])
        except:
            print("End of file")
    day = day + 1
    i = i + 24


# df_Anomalies list for adding column Anomaly with True and False
df_Anomalies = []
# df_thresholds list for adding column Thresholds with the corresponding threshold for that time for both weekdays and weekend
df_Thresholds = []
Date_Time = []
# Thresholds decided per hour for both weekdays and weekend seperatly
# These threshold are spesific to the respective hospital and should be reevaluated when used for another hospital
thresholds = [40, 40, 40, 30, 30, 30, 20, 60, 100, 120, 150, 130, 100, 110, 120, 120, 120, 100, 80, 60, 60, 40, 50, 50, 20, 20, 20, 20, 15, 15, 15, 30, 60, 80, 80, 80, 70, 80, 80, 90, 60, 50, 35, 25, 40, 30, 20, 20]
for threshold, hour_average in zip(thresholds, [hour_average00, hour_average01, hour_average02, hour_average03, hour_average04, hour_average05, hour_average06, hour_average07, hour_average08, hour_average09, hour_average10, hour_average11, hour_average12, hour_average13, hour_average14, hour_average15, hour_average16, hour_average17, hour_average18, hour_average19, hour_average20, hour_average21, hour_average22, hour_average23, weekend_hour_average00, weekend_hour_average01, weekend_hour_average02, weekend_hour_average03, weekend_hour_average04, weekend_hour_average05, weekend_hour_average06, weekend_hour_average07, weekend_hour_average08, weekend_hour_average09, weekend_hour_average10, weekend_hour_average11, weekend_hour_average12, weekend_hour_average13, weekend_hour_average14, weekend_hour_average15, weekend_hour_average16, weekend_hour_average17, weekend_hour_average18, weekend_hour_average19, weekend_hour_average20, weekend_hour_average21, weekend_hour_average22, weekend_hour_average23]):
    # anomalies list for plot
    anomalies = []
    th_counts = []
    temp_normal = []
    # get list of all Counts in this hour average list
    for elem in hour_average:
        th_counts.append(list(df['Count'][df.DateTime == elem])[0])
    # calculate what the top 0.3 (this can change depending on the number below) is
    # threshold = np.percentile(th_counts, 99.7)
    print(threshold)
    # for every data point cur_point is DateTime and th is Count
    for cur_point, th in zip(hour_average, th_counts):
        Date_Time.append(cur_point)
        df_Thresholds.append(threshold)
        if th >= threshold:
            # for plot
            anomalies.append(th)
            # for df column
            df_Anomalies.append(True)
            # for plot
            temp_normal.append(0)
        else:
             # for plot
            anomalies.append(0)
             # for column
            df_Anomalies.append(False)
             # for plot
            temp_normal.append(th)
    # per list a plot is shown with all datapoint, normal data points are blue
    # yellow points are anomalies accoprding to the code (may be manually altered later in the proces)
    # sns.regplot(x = list(range(1,len(hour_average)+1)), y = temp_normal, color = 'b')
    # sns.regplot(x = list(range(1,len(hour_average)+1)), y = anomalies, color = 'y')
    # plt.show()

# make dataframe to merge with the first dataframe so the columns Anomaly and threshold are added
df2 = pd.DataFrame({
    'DateTime': Date_Time,
    'Threshold': df_Thresholds,
    'Anomaly': df_Anomalies
})
new_df = pd.merge(df, df2, on='DateTime')
print(new_df)

# save new dataframe with column Anomaly in csv file
new_df.to_csv('labels2.csv')