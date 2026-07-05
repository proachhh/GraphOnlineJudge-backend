import torch
import torch.optim as optim
import numpy as np
from recommend.model import SimpleRecommender
import pickle

def roc_auc(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    tp, fp, auc_val, last_pred = 0, 0, 0, None
    for idx in order:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        if last_pred is not None and y_pred[idx] != last_pred:
            auc_val += tp * fp
            tp, fp = 0, 0
        last_pred = y_pred[idx]
    auc_val += tp * fp
    return auc_val / (pos * neg)

def train():
    with open('/data/recommend_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    num_users = data['num_users']
    num_items = data['num_items']
    
    train_u = torch.tensor([x[0] for x in X_train], dtype=torch.long)
    train_i = torch.tensor([x[1] for x in X_train], dtype=torch.long)
    train_y = torch.tensor(y_train, dtype=torch.float)
    test_u = torch.tensor([x[0] for x in X_test], dtype=torch.long)
    test_i = torch.tensor([x[1] for x in X_test], dtype=torch.long)
    test_y = torch.tensor(y_test, dtype=torch.float)
    
    model = SimpleRecommender(num_users, num_items, emb_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 512
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(len(train_u))
        total_loss = 0
        for i in range(0, len(train_u), batch_size):
            batch_idx = idx[i:i+batch_size]
            u = train_u[batch_idx]
            it = train_i[batch_idx]
            y = train_y[batch_idx]
            optimizer.zero_grad()
            pred = model(u, it)
            loss = -torch.mean(torch.log(pred + 1e-8)*y + torch.log(1-pred + 1e-8)*(1-y))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(train_u)//batch_size+1):.4f}")
        
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_test = model(test_u, test_i).numpy()
                auc = roc_auc(test_y.numpy(), pred_test)
                print(f"  Test AUC: {auc:.4f}")
    
    torch.save(model.state_dict(), '/data/recommend_model.pt')
    print("模型已保存")

if __name__ == '__main__':
    train()