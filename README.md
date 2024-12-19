import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("C:/Users/keert/Downloads/Task-1.xlsx")
print(df.head())
print(df.info())
print(df.columns)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
print(df.columns)
print(df[['year', 'male_population', 'female_population']].head())
print(df[['male_population', 'female_population']].dtypes)
df['male population'] = pd.to_numeric(df['male_population'], errors='coerce')
df['female population'] = pd.to_numeric(df['female_population'], errors='coerce')
print(df['male population'])
print(df['female population'])
print(df[['male population', 'female population']].describe())
plt.figure(figsize=(12, 6))

# Plot male and female population with dots after each year
plt.plot(df['year'], df['male_population'], label='Male Population', color='blue', marker='o', markersize=6)
plt.plot(df['year'], df['female_population'], label='Female Population', color='pink', marker='o', markersize=6)

# Add dots as annotations after each year
for x, y in zip(df['year'], df['male_population']):
    plt.text(x, y, '.', color='blue', fontsize=14, ha='center', va='bottom')  # Dot for Male Population

for x, y in zip(df['year'], df['female_population']):
    plt.text(x, y, '.', color='pink', fontsize=14, ha='center', va='bottom')  # Dot for Female Population

# Add title and axis labels
plt.title('Male vs Female Population Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=12)

# Add legend
plt.legend()

# Show plot
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='year', y='male_population', data=subset_years, color='blue', label='Male Population')
sns.barplot(x='year', y='female_population', data=subset_years, color='pink', label='Female Population')

plt.title('Gender-wise Population Distribution (1960-2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram for male and female populations
plt.figure(figsize=(12, 6))

# Plot histogram for male population with slightly shifted bins for overlap
sns.histplot(df['male population'], color='blue', label='Male Population', kde=True, bins=10, alpha=0.5)

# Plot histogram for female population with slightly shifted bins for overlap
sns.histplot(df['female population'], color='pink', label='Female Population', kde=True, bins=10, alpha=0.5)

# Add title and axis labels
plt.title('Distribution of Male and Female Populations (1960-2023)', fontsize=16)
plt.xlabel('Population', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Add legend
plt.legend()

# Show plot
plt.show()

