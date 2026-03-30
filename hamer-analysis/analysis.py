import pickle
import numpy as np

def frame_number(image_name):
    return int(image_name.split('_')[1].split('.')[0])

# open results from HaMeR
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"nb hands detected: {len(results)}", np.array(results).shape)
print(f"right hands: {len([i for i in results if i['is_right']])}")

print(results[0]['confidence'])

n_frames = len([i for i in results if i['is_right']])

# sort results by frame number
results_sorted = sorted(results, key=lambda r: frame_number(r['image_name']))

print([r['image_name'] for r in results_sorted[0:20]])

left_hands = [r for r in results_sorted if r['is_right'] == False]
right_hands = [r for r in results_sorted if r['is_right'] == True]

data = []
for i in range(n_frames):
    data.append( {'left': left_hands[i], 'right': right_hands[i]} )

print(data[0]['right']['confidence'])
