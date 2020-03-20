import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW

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

class Probe():
    def __init__(self, hidden_size, lr=3e-5, adam_epsilon=1e-8, max_grad_norm=1.0):
        self.hidden_size = hidden_size
        self.model_start_idx = SoftmaxRegression(hidden_size)
        self.model_end_idx = SoftmaxRegression(hidden_size)

        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        
        self.start_optimizer = AdamW(self.model_start_idx.parameters(), lr=self.lr, eps=self.adam_epsilon, correct_bias=False)
        self.end_optimizer = AdamW(self.model_end_idx.parameters(), lr=self.lr, eps=self.adam_epsilon, correct_bias=False)

    def train(self, inputs, start_targets, end_targets, device, weight=None):
        
        inputs = inputs.to(device)
        start_targets = start_targets.to(device)
        end_targets = end_targets.to(device)
        self.model_start_idx.to(device)
        self.model_end_idx.to(device)

        self.model_start_idx.train()
        self.model_end_idx.train()

        with torch.set_grad_enabled(True):
            batch_loss = 0

            start_scores = self.model_start_idx(inputs)
            end_scores = self.model_end_idx(inputs)
            
            ignored_index = start_scores.size(1)
            start_targets.clamp_(0, ignored_index)
            end_targets.clamp_(0, ignored_index)

            if weight is not None:
                start_weight = weight[0, :]
                end_weight = weight[1, :]
            else:
                start_weight = None
                end_weight = None

            start_loss = nn.CrossEntropyLoss(weight=start_weight, ignore_index=ignored_index)(start_scores, start_targets)
            end_loss = nn.CrossEntropyLoss(weight=end_weight, ignore_index=ignored_index)(end_scores, end_targets)
            loss = (start_loss+end_loss)/2.0

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model_start_idx.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model_end_idx.parameters(), self.max_grad_norm)
            
            self.start_optimizer.step()
            self.end_optimizer.step()

            self.model_start_idx.zero_grad()
            self.model_end_idx.zero_grad()

        return batch_loss

    def predict(self, inputs, device, threshold=1.0, context_start=1, context_end=None, max_answer_length=17):
        """ Function to predict the start and end endices in a question answer sequence
            inputs: tensor (batch_size, seq_len, hidden_size) are attention weighted hidden state outputs
            device: string ('cuda' or 'cpu') tells pytorch where to run computations
            threshold: float (e.g. 1.0 for no ans score*1.0) controlling tradeoff between answer and no answer prediction
            context_start: tensor (batch_size) can be specified to avoid selecting tokens in question e.g. 1 if only start token at 0
            contenxt_end: tensor (batch_size) can be specified to avoid looking at tokens in padding e.g. seq_len if full sequence
            max_answer_length: integer (e.g. 17 for longest in Squad2.0 by words not tokens) constraining search space by giving maximum answer length in tokens
        """

        # TODO: Make work over batches of greater than size 1

        _, seq_len, _ = inputs.shape
        if context_end == None:
            context_end = seq_len
        
        inputs = inputs.to(device)

        self.model_start_idx.to(device)
        self.model_end_idx.to(device)

        self.model_start_idx.eval()
        self.model_end_idx.eval()

        S = self.model_start_idx
        E = self.model_end_idx

        with torch.no_grad():
            start_scores = S(inputs)
            end_scores = E(inputs)

            start_null = start_scores[:, 0]
            end_null = end_scores[:, 0]
            score_null = start_null + end_null

            start_best, end_best = context_start, context_start
            score_best = start_scores[:, start_best] + end_scores[:, end_best]

            for start_curr in range(context_start, context_end):
                start_score = start_scores[:, start_curr]
                end_scores_valid = end_scores[:, start_curr:min(start_curr+max_answer_length+1, context_end)]
                end_score, end_idx = end_scores_valid.max(-1)
                end_curr = end_idx+start_curr
                score_curr = start_score + end_score
                if score_curr >= score_best:
                    score_best = score_curr
                    start_best, end_best = start_curr, end_curr

            non_null_more_likely_than_null = score_best >= (score_null+threshold)
            
            # Multiply by mask to force idx where null is more probable to zero
            score = non_null_more_likely_than_null*score_best+(~non_null_more_likely_than_null)*score_null
            start_idx = non_null_more_likely_than_null*start_best
            end_idx = non_null_more_likely_than_null*end_best

        return score.cpu().numpy(), start_idx.cpu().numpy(), end_idx.cpu().numpy()
    
    def save(self, probe_dir, layer):
        torch.save(self.model_start_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_start_idx")
        torch.save(self.model_end_idx.state_dict(), probe_dir + "/layer_" + str(layer) + "_end_idx")

    def load(self, probe_dir, layer, device):
        self.model_start_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_start_idx", map_location=device))
        self.model_end_idx.load_state_dict(torch.load(probe_dir + "/layer_" + str(layer) + "_end_idx", map_location=device))


if __name__ == "__main__":
    import os
    SEED = 1

    torch.manual_seed(SEED)

    batch_size = 3
    seq_len = 7
    hidden_size = 8

    inputs = torch.randn((batch_size, seq_len, hidden_size)) # matrix of size (batch_size, seq_len, hidden_size)
    start_idx_targets = torch.tensor([4, 0, 2]) # matrix of size (batch_size) where entry is class idx
    end_idx_targets = torch.tensor([6, 0, 3]) # matrix of size (batch_size) where entry is class idx

    max_epoch = 20000
    print_every = 1000
    epoch = 0
    device = "cpu"
    model = Probe(hidden_size)
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

    model = Probe(hidden_size)
    model.load(save_dir, 0, device)

    print("After load predict: ", (model.predict(inputs, device)))
