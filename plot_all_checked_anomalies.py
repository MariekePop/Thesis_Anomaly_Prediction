# makes a plot with all normal data points in blue and all anomalies in yellow

import pandas as pd
import matplotlib.pyplot as plt

# read data from csv file
df = pd.read_csv(r'C:\Users\f.de.kok\Documents\thesis\new_labels_with_thresholds.csv')
df2 = df.loc[df['y'] < 1494]
df3 = df2.loc[df['y'] >1002]

# plot per category in column Anomaly (first plot all False values and then all True values)
groups = df3.groupby('Anomaly')
for name, group in groups:
    # if we plot the group False thus normal data points label is 'Normal data points'
    if not name:
        plt.plot(group.Count, marker='o', linestyle='', markersize=4, label='Normal data points')
    # if we plot the group True thus anomalies label is 'Anomalies'
    else:
        plt.plot(group.Count, marker='o', linestyle='', markersize=4, label='Anomalies')

# add title, x and y labels and the legend
plt.title("Anomalies Within ChipSoft's Hospital Data")
plt.xlabel('Data points')
plt.ylabel('Number of JIPS (error messages)')
plt.legend()
plt.show()