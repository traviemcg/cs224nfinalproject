import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW

class SoftmaxRegression(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super(SoftmaxRegression, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.W = nn.Linear(self.hidden_size, 1, bias=True)
    
    def forward(self, input):
        scores = self.W(input).squeeze(-1)
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
    def __init__(self, seq_len, hidden_size):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.model_start_idx = SoftmaxRegression(seq_len, hidden_size)
        self.model_end_idx = SoftmaxRegression(seq_len, hidden_size)

        self.lr = 3e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0

    def train(self, inputs, start_targets, end_targets, device):
        
        start_model = self.model_start_idx.to(device)
        end_model = self.model_end_idx.to(device)

        start_optimizer = AdamW(start_model.parameters(), lr=self.lr, eps=self.adam_epsilon)
        end_optimizer = AdamW(end_model.parameters(), lr=self.lr, eps=self.adam_epsilon)
        
        start_model.train()
        end_model.train()

        with torch.set_grad_enabled(True):
            start_scores = start_model.forward(inputs)
            end_scores = end_model.forward(inputs)
            ignore_index = start_scores.size(1)
            start_targets.clamp_(0, ignore_index)
            end_targets.clamp_(0, ignore_index)

            start_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(start_scores, start_targets)
            end_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(end_scores, end_targets)
            loss = (start_loss+end_loss)/2.0
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(start_model.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(end_model.parameters(), self.max_grad_norm)
            
            start_optimizer.step()
            start_optimizer.zero_grad()
            end_optimizer.step()
            end_optimizer.zero_grad()

        return loss

    def predict(self, inputs, device):
        with torch.no_grad():
            start_idx = self.model_start_idx.to(device).eval().predict(inputs)
            end_idx = self.model_end_idx.to(device).eval().predict(inputs)

            start_scores = self.model_start_idx.to(device).eval().forward(inputs)
            end_scores = self.model_end_idx.to(device).eval().forward(inputs)
            
            threshold = 1000000
            mask = start_scores+end_scores >= threshold # boolean mask where >= threshold is 1
            
            start_masked = start_idx*mask
            end_masked = end_idx*mask

            idxs = torch.stack([start_masked.unsqueeze(-1), end_masked.unsqueeze(-1)], dim=-1)
            np_idxs = idxs.cpu().numpy()
        
        # Return (class_size, 2) array where both entries are 0 if is impossible
        # If want to extend to larger batches, let whole first index go
        return np_idxs[0, :, 0, :]
    
    def save(self, probe_dir, layer):
        torch.save(self.model_start_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_start_idx")
        torch.save(self.model_end_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_end_idx")

    def load(self, probe_dir, layer):
        self.model_start_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_start_idx"))
        self.model_end_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_end_idx"))

