import pickle
from collections import defaultdict

h_sets = defaultdict(list)
ws_sets = defaultdict(list)

for i in range(10):
    name_h = "pickles_new/new_h_"+str(i)+".pickle"
    name_ws = "pickles_new/new_ws_"+str(i)+".pickle"
    # name_h = "pickles_old/h_"+str(i)+".pickle"
    # name_ws = "pickles_old/ws_"+str(i)+".pickle"
    with open(name_h, 'rb') as f:
        h_sets[i] = pickle.load(f)
    with open(name_ws, 'rb') as f:
        ws_sets[i] = pickle.load(f)

h_sets_name = "pickles_new/h_sets"+".pickle"
ws_sets_name = "pickles_new/ws_sets"+".pickle"
# h_sets_name = "pickles_old/h_sets"+".pickle"
# ws_sets_name = "pickles_old/ws_sets"+".pickle"

with open(h_sets_name, 'wb') as f:
    pickle.dump(h_sets, f)
with open(ws_sets_name, 'wb') as f:
    pickle.dump(ws_sets, f)    