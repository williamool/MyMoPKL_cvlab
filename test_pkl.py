import pickle

with open("emb_train_ITSDT.pkl", "rb") as f:
    data = pickle.load(f)

one_key = list(data.keys())[0]
print("Key:", one_key)
print("Value type:", type(data[one_key]))
print("Value shape:", getattr(data[one_key], "shape", None))
print("Value example:", data[one_key][:5])  
