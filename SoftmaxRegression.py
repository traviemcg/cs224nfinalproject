import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

class SoftmaxRegression(nn.Module):
    def __init__(self, hidden_size):
        super(SoftmaxRegression, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(self.hidden_size, 1, bias=True)
        self.W.weight.data.normal_(mean=0.0, std=0.02)
        self.W.bias.data.zero_()
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, input):
        scores = self.W(self.dropout(input)).squeeze(-1)
        return scores
    
    def predict_proba(self, input):
        scores = self.forward(input)
        p = F.softmax(scores, dim=-1)
        return p
    
    def predict(self, input):
        p = self.predict_proba(input)
        _, preds = p.max(-1)
        return preds

class MultiSoftmaxRegression():
    def __init__(self, hidden_size, num_train_samples, num_train_epochs):
        self.hidden_size = hidden_size
        self.model_start_idx = SoftmaxRegression(hidden_size)
        self.model_end_idx = SoftmaxRegression(hidden_size)

        self.lr = 3e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0

        self.start_optimizer = AdamW(self.model_start_idx.parameters(), lr=self.lr, eps=self.adam_epsilon)
        self.end_optimizer = AdamW(self.model_end_idx .parameters(), lr=self.lr, eps=self.adam_epsilon)
        
        self.total_steps = num_train_samples * num_train_epochs
        self.start_scheduler = get_linear_schedule_with_warmup(self.start_optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)
        self.end_scheduler = get_linear_schedule_with_warmup(self.end_optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)

    def train(self, inputs, start_targets, end_targets, device):
        
        self.model_start_idx.to(device)
        self.model_end_idx.to(device)

        self.model_start_idx.train()
        self.model_end_idx.train()

        with torch.set_grad_enabled(True):
            start_scores = self.model_start_idx(inputs)
            end_scores = self.model_end_idx(inputs)
            
            ignored_index = start_scores.size(1)
            start_targets.clamp_(0, ignored_index)
            end_targets.clamp_(0, ignored_index)

            start_loss = nn.CrossEntropyLoss(ignore_index=ignored_index)(start_scores, start_targets)
            end_loss = nn.CrossEntropyLoss(ignore_index=ignored_index)(end_scores, end_targets)
            loss = (start_loss+end_loss)/2.0
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model_start_idx.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model_end_idx.parameters(), self.max_grad_norm)
            
            self.start_optimizer.step()
            self.end_optimizer.step()

            self.start_scheduler.step()
            self.end_scheduler.step()

            self.model_start_idx.zero_grad()
            self.model_end_idx.zero_grad()

        return loss

    def predict(self, inputs, device):

        self.model_start_idx.to(device)
        self.model_end_idx.to(device)

        self.model_start_idx.eval()
        self.model_end_idx.eval()

        with torch.no_grad():
            start_idx = self.model_start_idx.predict(inputs)
            end_idx = self.model_end_idx.predict(inputs)

        return start_idx.cpu().numpy(), end_idx.cpu().numpy()
    
    def save(self, probe_dir, layer):
        torch.save(self.model_start_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_start_idx")
        torch.save(self.model_end_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_end_idx")

    def load(self, probe_dir, layer):
        self.model_start_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_start_idx"))
        self.model_end_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_end_idx"))


if __name__ == "__main__":
    import os
    SEED = 1

    torch.manual_seed(SEED)

    batch_size = 3
    seq_len = 4
    hidden_size = 5

    inputs = torch.randn((batch_size, seq_len, hidden_size)) # matrix of size (batch_size, seq_len, hidden_size)
    start_idx_targets = torch.tensor([0, 0, 2]) # matrix of size (batch_size) where entry is class idx
    end_idx_targets = torch.tensor([1, 0, 3]) # matrix of size (batch_size) where entry is class idx

    max_epoch = 10000
    print_every = 500
    epoch = 0
    device = "cpu"
    model = MultiSoftmaxRegression(hidden_size, batch_size, max_epoch)
    while epoch < max_epoch:
        epoch += 1
        loss = model.train(inputs, start_idx_targets, end_idx_targets, device)
        
        if epoch%print_every==0:
            print("epoch {}, loss {:.2f}".format(epoch, loss))

    print("Before load predict: ", (model.predict(inputs, device)))

    save_dir = "./sm_model"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model.save(save_dir, 0)

    model = MultiSoftmaxRegression(hidden_size, batch_size, max_epoch)
    model.load(save_dir, 0)

    print("After load predict: ", (model.predict(inputs, device)))
