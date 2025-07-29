import numpy as np
npz = np.load("lstm_sequences.npz")
print(npz["timestamps"][-10:])  # zeigt die letzten 10 gültigen