import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW

class IdxSoftmaxRegression(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super(IdxSoftmaxRegression, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.W = nn.Linear(self.hidden_size, 1, bias=True)
    
    def forward(self, input):
        scores = self.W(input).squeeze(-1)
        return scores
    
    def train_forward(self, input, target, device):
        scores = self.forward(input)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(scores, target)
        return loss
    
    def predict_proba(self, input):
        scores = self.forward(input)
        p = F.softmax(scores, dim=-1)
        return p
    
    def predict(self, input):
        p = self.predict_proba(input)
        _, preds = p.max(-1)
        return preds

class ImpossibleSoftmaxRegression(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super(ImpossibleSoftmaxRegression, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.W = nn.Linear(self.seq_len*self.hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        input_flattened = input.flatten(-2, -1)
        scores = self.W(input_flattened).squeeze(-1)
        p = self.sigmoid(scores)
        return p
    
    def train_forward(self, input, target, device):
        p = self.forward(input)
        # 1/3 the questions are impossible, 2/3 are not impossible. So weight them
        if target[0] == 1:
            weights = torch.tensor([2.0]).unsqueeze(0).to(device)
        elif target[0] == 0:
            weights = torch.tensor([1.0]).unsqueeze(0).to(device)
        loss = nn.BCELoss(weight=weights)(p, target)
        return loss
    
    def predict_proba(self, input):
        p = self.forward(input)
        return p
    
    def predict(self, input):
        p = self.predict_proba(input)
        preds = torch.round(p)
        return preds

class MultiSoftmaxRegression():
    def __init__(self, seq_len, hidden_size):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.model_start_idx = IdxSoftmaxRegression(seq_len, hidden_size)
        self.model_stop_idx = IdxSoftmaxRegression(seq_len, hidden_size)
        self.model_is_impossible = ImpossibleSoftmaxRegression(seq_len, hidden_size)

        self.lr = 3e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0

    def train_step(self, inputs, targets, mode, device):
        
        if mode == "start":
            model = self.model_start_idx.to(device)
        elif mode == "stop":
            model = self.model_stop_idx.to(device)
        elif mode == "impossible":
            model = self.model_is_impossible.to(device)
        else:
            model = None
            assert model != None, "Invalid value mode={}, should be 'start', 'stop', or 'impossible'!".format(mode)

        optimizer = AdamW(model.parameters(), lr=self.lr, eps=self.adam_epsilon)
        model.train()
        with torch.set_grad_enabled(True):
            
            loss = model.train_forward(inputs, targets, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def train(self, attentions, is_imp, start, stop, device):
        self.train_step(attentions, is_imp, 'impossible', device)
        if not is_imp[0]:
            self.train_step(attentions, start, 'start', device)
            self.train_step(attentions, stop, 'stop', device)


    def predict(self, inputs, device):
        with torch.no_grad():
            start_idx = self.model_start_idx.to(device).eval().predict(inputs)
            stop_idx = self.model_stop_idx.to(device).eval().predict(inputs)
            

            # code when using the ImpossibleSoftmaxRegression model
            """
            #
            is_impossible = self.model_is_impossible.to(device).eval().predict(inputs)
            start_idx = start_idx+1 
            stop_idx = stop_idx+1

            is_ans = 1-is_impossible
            #
            """

            # code when using threshold
            #
            threshold = 10/384*2
            start_idx_proba = self.model_start_idx.to(device).eval().predict_proba(inputs)
            stop_idx_proba = self.model_stop_idx.to(device).eval().predict_proba(inputs)
            is_ans = (start_idx_proba+stop_idx_proba) >= threshold
            #

            masked_start = start_idx*is_ans-1
            masked_stop = stop_idx*is_ans-1


            idxs = torch.stack([masked_start.unsqueeze(-1), masked_stop.unsqueeze(-1)], dim=-1)
            np_idxs = idxs.cpu().numpy()
        
        # Return (batch_size, 2) array where both entries are -1 if is_impossible==1
        return np_idxs[:, 0, :]
    
    def save(self, probe_dir, layer):
        torch.save(self.model_start_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_start_idx")
        torch.save(self.model_stop_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_stop_idx")
        torch.save(self.model_is_impossible.state_dict(), probe_dir + "/layer_" + str(layer) + "_is_impossible")

    def load(self, probe_dir, layer):
        self.model_start_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_start_idx"))
        self.model_stop_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_stop_idx"))
        self.model_is_impossible.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_is_impossible"))

