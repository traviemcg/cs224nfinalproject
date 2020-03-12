import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW

class SoftmaxRegression(nn.Module):
    def __init__(self, hidden_size):
        super(SoftmaxRegression, self).__init__()
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
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.model_start_idx = SoftmaxRegression(hidden_size)
        self.model_end_idx = SoftmaxRegression(hidden_size)

        self.lr = 3e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0

    def train(self, inputs, start_targets, end_targets, device):
        
        start_model = self.model_start_idx.to(device)
        end_model = self.model_end_idx.to(device)

        start_model.train()
        end_model.train()

        start_optimizer = AdamW(start_model.parameters(), lr=self.lr, eps=self.adam_epsilon)
        end_optimizer = AdamW(end_model.parameters(), lr=self.lr, eps=self.adam_epsilon)

        with torch.set_grad_enabled(True):
            start_scores = start_model(inputs)
            end_scores = end_model(inputs)
            
            ignored_index = start_scores.size(1)
            start_targets.clamp_(0, ignored_index)
            end_targets.clamp_(0, ignored_index)

            start_loss = nn.CrossEntropyLoss(ignore_index=ignored_index)(start_scores, start_targets)
            end_loss = nn.CrossEntropyLoss(ignore_index=ignored_index)(end_scores, end_targets)
            loss = (start_loss+end_loss)/2.0
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(start_model.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(end_model.parameters(), self.max_grad_norm)
            
            start_optimizer.step()
            end_optimizer.step()

            start_model.zero_grad()
            end_model.zero_grad()

        return loss

    def predict(self, inputs, device):

        start_model = self.model_start_idx.to(device)
        end_model = self.model_end_idx.to(device)

        start_model.eval()
        end_model.eval()

        with torch.no_grad():
            start_idx = start_model.predict(inputs)
            end_idx = end_model.predict(inputs)

        return start_idx.cpu().numpy(), end_idx.cpu().numpy()
    
    def save(self, probe_dir, layer):
        torch.save(self.model_start_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_start_idx")
        torch.save(self.model_end_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_end_idx")

    def load(self, probe_dir, layer):
        self.model_start_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_start_idx"))
        self.model_end_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_end_idx"))

