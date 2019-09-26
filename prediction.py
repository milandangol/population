import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
csv_file = 'pop.csv'
df= pd.read_csv(csv_file, skiprows = 4)
remove = [ 'Country Code', 'Indicator Name', 'Indicator Code','Unnamed: 63']
df.drop(columns=remove, inplace=True)
df.rename(columns={'Country Name': 'Country'}, inplace=True)
print(df[df.isnull().T.any().T])
df.dropna(inplace=True)
record = df[df.index == 'Australia']
record.columns = record.columns.astype(str)
years = record.columns.tolist()
population = record.values.tolist()[0]
plt.scatter(years, population)
plt.plot(years, population)
plt.xticks(rotation='vertical') 
plt.title('Australia\'s future population from 1960 to 2028')
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.show()
countries = df['Country'].tolist()
temp_df = pd.Dataframe()
record = df[df['Country'] == country].drop(['Country'], axis=1)
record = record.T
record.reset_index(inplace=True)
record.columns = ['Year', 'Population']
X = record['Year']
Y = record['Population']
regressor = LinearRegression()
regressor.fit(np.array(X).reshape(-1,1), Y)
for year in range(2018,2029):
        future_population = round(regressor.predict(np.array([year]).reshape(-1,1))[0])
        row = pd.DataFrame([[year,future_population]], columns=['Year','Population'])
        record = record.append(row, ignore_index=True)
record = record.T
new_header = record.iloc[0]
record = record[1:]
record.columns = new_header
record.columns.name = None
record.index = [country]
temp_df = pd.concat([temp_df, record])
df = temp_df
df.to_csv('future_world_population.csv')
