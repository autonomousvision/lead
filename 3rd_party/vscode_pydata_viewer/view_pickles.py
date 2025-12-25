import sys
import pickle
import lzma
import json
import numpy as np

def process_pickle_file(file_path):
    with lzma.open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(json.dumps(data, indent=4, default=_json_default))

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)

if __name__ == "__main__":
    file_type = int(sys.argv[1])
    file_path = sys.argv[2]

    if file_type == 1:  # Pickle file
        process_pickle_file(file_path)
