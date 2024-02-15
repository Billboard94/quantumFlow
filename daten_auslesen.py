import pandas as pd
from io import StringIO


# Daten als DataFrame einlesen
df = pd.read_csv('trainingdata_small.csv', sep=";", decimal=",")
#df = pd.read_csv('trainingdata_small.csv', header=0)

# Einzelne Elemente ausgeben
# Beispiel: Wert in der ersten Zeile und der Spalte 'L_chi'
wert = df.at[0, 'L_chi']
print("Wert in der ersten Zeile und der Spalte 'L_chi':", wert)

# Beispiel: Wert in der dritten Zeile und der sechsten Spalte ('U')
wert = df.at[0, 'rho']
print("Wert in der dritten Zeile und der sechsten Spalte ('U'):", wert)

# Index der Spalte 'rho' finden
rho_index = df.columns.get_loc('rho')

# Alle Werte in den Spalten nach 'rho' ausgeben
werte_nach_rho = df.iloc[:, rho_index :]
print("Werte nach 'rho':\n", werte_nach_rho)

# Alle Werte in den Spalten nach 'rho' ausgeben
werte_nach_rho = df.iloc[0, 6 :10]
print("Werte nach 'rho':\n", werte_nach_rho)
