import sys
from janitor.functions import read_commandline

for x in sys.argv:
     print("Argument: ", x)
df = read_commandline()
print(df)
