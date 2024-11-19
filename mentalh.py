import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
# Wczytywanie oraz przygotowanie danych z CSV i połączenie z bazą danych SQLite
mentalh = pd.read_csv(r"C:\Users\zales\Documents\projekty\mental\mental_health_diagnosis_treatment_.csv")

print("Brakujące wartości w każdej kolumnie:")
print(mentalh.isnull().sum())
numeric_cols = mentalh.select_dtypes(include=['number']).columns

mentalh[numeric_cols] = mentalh[numeric_cols].fillna(mentalh[numeric_cols].mean())
categorical_cols = mentalh.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mentalh[col] = mentalh[col].fillna(mentalh[col].mode()[0])

scaler = StandardScaler()
mentalh[numeric_cols]=scaler.fit_transform(mentalh[numeric_cols])
print("Dane po skalowaniu")
print(mentalh.head())
conn = sqlite3.connect('mentalh.db')
cursor = conn.cursor()

# Wczytywanie danych do bazy danych SQLite
mentalh.to_sql('mentalh', conn, if_exists='replace', index=False)
# CZ 1:

# ŚREDNIE
cursor.execute("DROP TABLE IF EXISTS diagnosis_avg;")
cursor.execute('''
    CREATE TABLE diagnosis_avg (
        diagnosis TEXT,
        avg_symptom_severity REAL DEFAULT 0.0,
        avg_sleep_quality REAL DEFAULT 0.0,
        avg_mood_score REAL DEFAULT 0.0,
        avg_phy_activity REAL DEFAULT 0.0,
        avg_stress_level REAL DEFAULT 0.0,
        avg_age REAL DEFAULT 0.0
    );
''')
conn.commit()

query = '''
    SELECT 
        diagnosis,
        COALESCE(AVG("Symptom Severity (1-10)"), 0.0) AS avg_symptom_severity,
        COALESCE(AVG("Sleep Quality (1-10)"), 0.0) AS avg_sleep_quality,
        COALESCE(AVG("Mood Score (1-10)"), 0.0) AS avg_mood_score,
        COALESCE(AVG("Physical Activity (hrs/week)"), 0.0) AS avg_phy_activity,
        COALESCE(AVG("Stress Level (1-10)"), 0.0) AS avg_stress_level,
        COALESCE(AVG("Age"), 0.0) AS avg_age
    FROM mentalh
    GROUP BY diagnosis;
'''
cursor.execute(query)
results = cursor.fetchall()


for row in results:
    cursor.execute('''
        INSERT INTO diagnosis_avg (diagnosis, avg_symptom_severity, avg_sleep_quality, avg_mood_score, avg_phy_activity, avg_stress_level, avg_age)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', row)
conn.commit()


df_avg = pd.read_sql("SELECT * FROM diagnosis_avg", conn)
print("Tabela wyników średnich wartości dla diagnoz:")
print(df_avg)

# Płeć
cursor.execute("DROP TABLE IF EXISTS diagnosis_gender;")
cursor.execute('''CREATE TABLE diagnosis_gender (diagnosis TEXT, gender TEXT, count INTREGER);''')
               
query = '''
    SELECT "diagnosis", "gender", COUNT(*) AS count
    FROM mentalh
    GROUP BY "diagnosis", "gender";
'''
results_gender = cursor.execute(query).fetchall()

for row in results_gender:
    cursor.execute(''' INSERT INTO diagnosis_gender (diagnosis, gender, count) VALUES (?,?,?)''', (row[0], row[1], row[2]))

conn.commit

df_gender = pd.read_sql("SELECT * FROM diagnosis_gender ORDER BY count DESC", conn)
print("PŁEĆ A DIAGNOZA")
print(df_gender)

# LEKI

cursor.execute("DROP TABLE IF EXISTS diagnosis_med;")
cursor.execute(''' CREATE TABLE diagnosis_med (diagnosis TEXT, medication TEXT, count INTREGER); ''')

query = '''SELECT "diagnosis", "medication", COUNT (*) AS count FROM mentalh 
GROUP BY "diagnosis", "medication"; '''

results_med = cursor.execute(query).fetchall()

for row in results_med:
    cursor.execute('''INSERT INTO diagnosis_med (diagnosis,medication,count) VALUES (?,?,?)''', (row[0],row[1],row[2]))
conn.commit
df_med = pd.read_sql("SELECT *  FROM diagnosis_med ORDER BY count DESC", conn)
print("DIAGNOZA - LEK")
print(df_med)

# Rodzaj terapii 

cursor.execute("DROP TABLE IF EXISTS diagnosis_therapy")
cursor.execute('''CREATE TABLE diagnosis_therapy (diagnosis TEXT, "therapy type" TEXT, count INTREGER);''')
query = '''SELECT "diagnosis", "therapy type", COUNT (*) AS count 
FROM mentalh GROUP BY "diagnosis", "therapy type"; '''

results_therapy = cursor.execute(query).fetchall()

for row in results_therapy:
    cursor.execute('''INSERT INTO diagnosis_therapy (diagnosis, "therapy type", count) VALUES (?,?,?)''', (row[0], row[1], row[2]))
conn.commit
df_therapy= pd.read_sql("SELECT * FROM diagnosis_therapy ORDER BY count DESC", conn)
print("Typ terapii = diagnoza")
print(df_therapy)

#WYKRESY 
sns.set_palette("Blues")

sns.barplot(x='diagnosis', y='avg_symptom_severity', data=df_avg)
plt.xticks(rotation=90,ha='right', fontsize=7)
plt.title("Średnie nasilenie objawów wg diagnozy")
plt.show()

sns.barplot(x='diagnosis', y='count', hue='gender', data=df_gender)
plt.xticks(rotation=90,ha='right', fontsize=7)
plt.title("Płeć a diagnoza")
plt.show()

sns.barplot(x='diagnosis', y='count', hue='medication', data=df_med)
plt.xticks(rotation=90,ha='right', fontsize=7)
plt.title("Diagnoza i stosowane leki")
plt.show()

sns.barplot(x='diagnosis', y='count', hue='therapy type', data=df_therapy)
plt.xticks(rotation=90,ha='right', fontsize=7)
plt.title("Diagnoza i stosowane terapie")
plt.show()

selected_columns = ['Mood Score (1-10)', 'Symptom Severity (1-10)', 'Stress Level (1-10)', 'Age']
sns.pairplot(mentalh[selected_columns])
plt.suptitle("Relacje między wybranymi zmiennymi", y=1.02,fontsize =16)
plt.show()

sns.pairplot(mentalh[selected_columns + ['Diagnosis']], hue = 'Diagnosis')
plt.suptitle("Relacje z podziałem na diagnozy", y=1.02, fontsize = 16)
plt.show()
 
#CZ 2:
df = pd.read_sql("SELECT * FROM mentalh", conn)
label_encoder = LabelEncoder()
for col in ['Gender', 'Diagnosis', 'Therapy Type']:
    df[col] = label_encoder.fit_transform(df[col])

df['Treatment Effectiveness'] = (df['Mood Score (1-10)'] - df['Symptom Severity (1-10)'] - df['Stress Level (1-10)']) / 3

df['Treatment Effectiveness Interactions'] = (df['Mood Score (1-10)'] * df['Symptom Severity (1-10)']) / (df['Stress Level (1-10)'] + 1)

df['Adherence Index'] = (df['Adherence to Treatment (%)']* df['Treatment Progress (1-10)'])

df['Mood Improvement Ratio'] = df['Mood Score (1-10)'] / (df['Symptom Severity (1-10)'] + 1)

df['Gender Therapy Interaction'] = df['Gender'] * df['Therapy Type']

# CZ 3 drzewa losowe:
X = df[['Gender', 'Diagnosis', 'Therapy Type', 'Mood Score (1-10)', 
        'Symptom Severity (1-10)', 'Stress Level (1-10)', 'Adherence to Treatment (%)', 
        'Treatment Progress (1-10)', 'Treatment Effectiveness', 'Adherence Index']]
y = df['Outcome']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

# korelacje
column_to_exclude = 'Patient ID'  # Zmień nazwę na kolumnę, którą chcesz pominąć
if column_to_exclude in df.columns:
    df_without_column = df.drop(columns=[column_to_exclude])
else:
    print(f"Kolumna '{column_to_exclude}' nie istnieje w DataFrame.")
numeric_df=df_without_column.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr()
print(correlation_matrix)

sns.set_theme(style='dark')
ax = sns.heatmap(correlation_matrix,cmap='PuBu',center=0,square=True,linewidths=5)
ax.set_title('Macierz Korelacji', fontsize=16, pad=15)
plt.show()
# Zamknięcie połączenia
conn.close()
