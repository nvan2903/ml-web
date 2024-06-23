a = [1,2,3,4]
import pandas as pd
data = pd.DataFrame(a, columns=['id'])
print(data[:, 0])