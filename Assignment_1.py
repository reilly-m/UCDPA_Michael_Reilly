#Import Rainfall Data 1850 to 2010
#Import package
from urllib.request import urlretrieve
import zipfile
import pandas as pd
import numpy as np
url = 'https://www.met.ie/cms/assets/uploads/2018/01/Long-Term-IIP-network-1.zip'
urlretrieve(url, 'Long-Term-IIP-network-1.zip')
import zipfile
with zipfile.ZipFile('Long-Term-IIP-network-1.zip', 'r') as my_zip:
    my_zip.extract('IIP_National series.csv')
df_rain = pd.read_csv('IIP_National series.csv', skiprows = 2, index_col = 0)
#print(df_rain.head(161))

# Calculate Total Annual Rainfall
df_rain['Annual_Total'] = df_rain.iloc[:,:].sum(axis=1)
print(df_rain.head(161))

# Calculate Total Summer Rainfall (June / July / August)
df_rain['Summer_Total'] = df_rain.iloc[:,5:8].sum(axis=1)
print(df_rain.head(161))

# Wettest Year
column = df_rain["Annual_Total"]
max_value = column.max()
max_index = column.idxmax()
print('Wettest Year ' + str(max_index) + '  Total Rainfall that year ' + str(max_value))

# Wettest Summer
column = df_rain["Summer_Total"]
max_value = column.max()
max_index = column.idxmax()
print('Wettest Summer ' + str(max_index) + '  Total Rainfall that summer ' + str(max_value))

# Top 10 Wettest Years
df_rain_totals = df_rain[['Annual_Total']]
sorted_df = df_rain_totals.sort_values(by='Annual_Total' , ascending=False )
print(sorted_df.iloc[0:10, 0:1])

# Top 10 Wettest Summers
df_rain_totals = df_rain[['Summer_Total']]
sorted_df = df_rain_totals.sort_values(by='Summer_Total' , ascending=False )
print(sorted_df.iloc[0:10,0:1])

# Inter Quartile Range
import numpy as np
q75, q25 = np.percentile(df_rain['Annual_Total'], [75,25])
iqr = q75 - q25
print("Inter Quartile Range  " + str(iqr))

# Box Plot Total Rain
import matplotlib.pyplot as plt
column = df_rain["Annual_Total"]
plt.boxplot(column)
plt.show()


# Graph Total Annual Rainfall 1850 - 2010
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_rain = df_rain.reset_index()
x = df_rain['Year'].head(161)
y = df_rain['Annual_Total'].head(161)
plt.plot(x,y, marker="o", linestyle="-", color="b", label="rainfall")
plt.title("Annual Rainfall 1850 - 2010")
plt.xlabel("Year")
plt.ylabel("Total Rainfall in mm")
plt.legend(['Annual Rainfall'])
plt.show()


# Graph Summer Rainfall
x = df_rain['Year'].head(161)
y = df_rain['Summer_Total'].head(161)
plt.plot(x,y, marker="o", linestyle="-", color="g", label="rainfall")
plt.title("Summer Rainfall 1850 - 2010")
plt.xlabel("Year")
plt.ylabel("Total Summer Rainfall in mm")
plt.legend(['Summer  Rainfall'])
plt.show()


# Calculate 3 Year Moving Average Annual Rainfall 1850 - 2010
df_rain_totals = df_rain[['Year','Annual_Total']]
print(df_rain_totals)
for i in range(0,df_rain_totals.shape[0]-2):
    df_rain_totals.loc[df_rain_totals.index[i + 2], 'SMA_3'] = np.round(((df_rain_totals.iloc[i, 1] + df_rain_totals.iloc[i + 1, 1] + df_rain_totals.iloc[i + 2, 1]) / 3), 1)
print(df_rain_totals)

# Calculate 10 Year Moving Average Annual Rainfall 1850 - 2010
for i in range(0,df_rain_totals.shape[0]-9):
    df_rain_totals.loc[df_rain_totals.index[i + 9], 'SMA_10'] = np.round(((df_rain_totals.iloc[i, 1] + df_rain_totals.iloc[i + 1, 1] + df_rain_totals.iloc[i + 2, 1] + df_rain_totals.iloc[i + 3, 1] + df_rain_totals.iloc[i + 4, 1]+ df_rain_totals.iloc[i + 5, 1] + df_rain_totals.iloc[i + 6, 1] + df_rain_totals.iloc[i + 7, 1] + df_rain_totals.iloc[i + 8, 1] + df_rain_totals.iloc[i + 9, 1]) / 10), 1)
print(df_rain_totals)

# Calculate 30 Year Moving Average Annual Rainfall 1850 - 2010
for i in range(0,df_rain_totals.shape[0]-29):
    df_rain_totals.loc[df_rain_totals.index[i + 9], 'SMA_30'] = np.round(((df_rain_totals.iloc[i, 1] + df_rain_totals.iloc[i + 1, 1] + df_rain_totals.iloc[i + 2, 1] + df_rain_totals.iloc[i + 3, 1] + df_rain_totals.iloc[i + 4, 1]+ df_rain_totals.iloc[i + 5, 1] + df_rain_totals.iloc[i + 6, 1] + df_rain_totals.iloc[i + 7, 1] + df_rain_totals.iloc[i + 8, 1] + df_rain_totals.iloc[i + 9, 1] +
                                                                           df_rain_totals.iloc[i + 10, 1] + df_rain_totals.iloc[i + 11, 1] + df_rain_totals.iloc[i + 12, 1] + df_rain_totals.iloc[i + 13, 1] + df_rain_totals.iloc[i + 14, 1] + df_rain_totals.iloc[i + 15, 1] + df_rain_totals.iloc[i + 16, 1] + df_rain_totals.iloc[i + 17, 1] + df_rain_totals.iloc[i + 18, 1] + df_rain_totals.iloc[i + 19, 1] +
                                                                           df_rain_totals.iloc[i + 20, 1] + df_rain_totals.iloc[i + 21, 1] + df_rain_totals.iloc[i + 22, 1] + df_rain_totals.iloc[i + 23, 1] + df_rain_totals.iloc[i + 24, 1] + df_rain_totals.iloc[i + 25, 1] + df_rain_totals.iloc[i + 26, 1] + df_rain_totals.iloc[i + 27, 1] + df_rain_totals.iloc[i + 28, 1] + df_rain_totals.iloc[i + 29, 1]) / 30), 1)
print(df_rain_totals)

# Plot 3 Year Moving Average
x = df_rain_totals['Year'].head(161)
y = df_rain_totals['SMA_3'].head(161)
plt.plot(x,y, marker="*", linestyle="-", color="r", label="")
plt.title("3 Year Moving Average Rainfall 1850 - 2010")
plt.xlabel("Year")
plt.ylabel("Average in mm")
plt.legend(['3 Yr SMA'])
plt.show()

# Plot 10 Year Moving Average
x = df_rain_totals['Year'].head(161)
y = df_rain_totals['SMA_10'].head(161)
plt.plot(x,y, marker=".", linestyle="-", color="c", label="")
plt.title("10 Year Moving Average Rainfall 1850 - 2010")
plt.xlabel("Year")
plt.ylabel("Average in mm")
plt.legend(['10 Yr SMA'])
plt.show()

#Plot 30 Year Moving Average
x = df_rain_totals['Year'].head(161)
y = df_rain_totals['SMA_30'].head(161)
plt.plot(x,y, marker="h", linestyle="-", color="m", label="")
plt.title("30 Year Moving Average Rainfall 1850 - 2010")
plt.xlabel("Year")
plt.ylabel("Average in mm")
plt.legend(['30 Yr SMA'])
plt.show()



# plot all 4 time Series
x = df_rain_totals['Year'].head(161)
y1 = df_rain_totals['Annual_Total'].head(161)
y2 = df_rain_totals['SMA_3'].head(161)
y3 = df_rain_totals['SMA_10'].head(161)
y4 = df_rain_totals['SMA_30'].head(161)
plt.plot(x,y1, marker="o", linestyle="solid", color="b", label="rainfall")
plt.plot(x,y2,marker="*", linestyle="dotted", color="r", label='rainfall')
plt.plot(x,y3,marker=".", linestyle="dashed", color="c", label='rainfall')
plt.plot(x,y4,marker="h", linestyle="dashdot", color="m", label='rainfall')
plt.title("Rainfall Time Series 1850 - 2010")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall in mm")
plt.legend()
plt.show()



