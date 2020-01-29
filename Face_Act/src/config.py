"""
Jan 2020
Xinru Yan
"""
train_file_path = '../data/296_face_train.csv'
dev_file_path = '../data/296_face_dev.csv'
test_file_path = '../data/296_face_test.csv'
test_result_path = '../result/'
save_best = '../model/model.pt'
#pte_path = '../data/GoogleNews-vectors-negative300.txt'
pte_path = None
emb_dim = 300
hidden_size = 64
hidden_layers = 2
batch_size = 64
lr = 0.1 #0.1
dropout = 0.5
max_epochs = 20
most_frequent_pte = 300000
# random seed
random_seed = 546237890
output_size = 1
setting = 'BiLSTM'
bidirectional = True