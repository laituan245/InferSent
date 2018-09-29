import torch
import json

# Helper functions
def read_json(fn):
    with open(fn) as f:
        data = json.load(f)
    return data

# Retrieve all captions from train_data.json and test_data.json
all_captions = []
train_data = read_json('pascal-sentences-dataset/train_data.json')
test_data = read_json('pascal-sentences-dataset/test_data.json')
all_data = train_data + test_data
for image_path, semantic_label, captions in all_data:
    all_captions = all_captions + captions
print('Retrieve all the captions to be encoded')

# Load our pre-trained model (in encoder/)
from models import InferSent
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
print('Load our pre-trained model (in encoder/)')

# Set word vector path for the model
W2V_PATH = 'fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)
print('Set word vector path for the model')

# Build the vocabulary of word vectors (i.e keep only those needed)
infersent.build_vocab(all_captions, tokenize=True)
print('Build the vocabulary of word vectors')

# Start encoding captions
caption2id = {}
f = open('pascal-sentences-dataset/text_features.txt', 'w+')
for caption in all_captions:
    current_feature = list(infersent.encode([caption], tokenize=True).squeeze())
    if not caption in caption2id: caption2id[caption] = 'caption_' + str(len(caption2id))
    current_feature = [str(feature) for feature in current_feature]
    current_feature_str = ' '.join(current_feature)
    f.write('%s %s\n' % (caption2id[caption], current_feature_str))
f.close()

with open('pascal-sentences-dataset/caption2id.json', 'w') as outfile:
    json.dump(caption2id, outfile)
