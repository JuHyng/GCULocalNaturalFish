import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, args, device):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        
        self.total_loss = 0.0
        self.all_predictions = []
        self.all_labels = []
        
        self.optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.train_loader) * args.epochs)

    def multi_label_metrics(self, predictions, labels):
        y_pred = np.where(predictions >= 0.5, 1, 0)
        y_true = labels

        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)

        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics

    def compute_metrics(self):
        all_logits = torch.cat(self.all_predictions, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)

        probs = torch.sigmoid(all_logits).cpu().numpy()
        labels = all_labels.cpu().numpy()

        return self.multi_label_metrics(predictions=probs, labels=labels)
    
    def clear_history(self):
        self.total_loss = 0.0
        self.all_predictions.clear()
        self.all_labels.clear()
        
    def run_epoch(self, dataloader, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.total_loss = 0.0
        self.all_predictions.clear()
        self.all_labels.clear()
        
        for batch_index, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = batch
            
            inputs = input_ids.to(self.device)
            attention_masks = attention_mask.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            decoder_attention_mask = decoder_attention_mask.to(self.device)
            
            labels = decoder_attention_mask[:, 1:].reshape(-1)
            
            # Inference
            loss, logits = self.model(inputs, attention_masks, decoder_input_ids, decoder_attention_mask)
        
            self.total_loss += loss.item()
            self.all_predictions.append(logits.cpu().detach())
            self.all_labels.append(labels.cpu().detach())
            
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        avg_loss = self.total_loss / len(dataloader)

        if mode in ['val', 'test']:
            metrics = self.compute_metrics()
            metrics['avg_loss'] = avg_loss
            return metrics   
        
        return {"loss": avg_loss}

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
