import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import CLIPModel
from transformers import CLIPProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import random
from colorama import Fore, Style
from tqdm import tqdm
from adabelief_pytorch import AdaBelief
from torch.nn.utils import clip_grad_norm_

training_losses = []
validation_losses = []
print("\n")

# Save training plots with matplotlib to:
plots_folder = 'ft-plots'
os.makedirs(plots_folder, exist_ok=True)

# Save model .pt files to: 
ft_checkpoints_folder = 'ft-checkpoints'
os.makedirs(ft_checkpoints_folder, exist_ok=True)

# Save verbose text / training logs to:
text_logs_folder = 'ft-logs'
os.makedirs(text_logs_folder, exist_ok=True)

def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    if epoch < adjust_after:
        return 1  # Initial slower unfreeze rate
    else:
        return increase_rate  # Increased rate after initial pass

def unfreeze_layers(model, epoch, total_layers=32, unfreeze_all=False):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
        layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'vision_model' in name:
                if i >= total_layers - layers_to_unfreeze:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

def monitor_gradient_norms(gradient_norms, threshold=1e-5):
    alert_messages = []
    for name, norms in gradient_norms.items():
        mean_norm = sum(norms) / len(norms)
        if mean_norm < threshold:  # Vanishing gradient
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 1000:  # Exploding gradient
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    if alert_messages:
        for message in alert_messages:
            print(message)
        # Optionally, you could also implement some automatic adjustment strategies here

def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    try:
        plt.figure(figsize=(20, 10))

    except:
        print(f"Error in plotting gradient norms: {e}")
        with open(f"{plots_folder}/gradient_norms_epoch_{epoch}.json", 'w') as f:
            json.dump(gradient_norms, f)
    
    # Choose a colormap
    cmap = plt.get_cmap('Spectral')
    
    # Sort the layers by the maximum gradient norm value, descending
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    
    # Generate distinct colors from the colormap
    colors = cmap(range(len(sorted_layers)))
    
    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    # Adjust legend: position at top right with smaller font size
    plt.legend(loc='upper right', fontsize='small')
    
    if use_log_scale:
        plt.yscale('log')
        plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}_log.png")
    else:
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    
    plt.close()

def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epochs_x):
        plt.plot(epochs_x, training_losses, label='Training Loss')
    if len(validation_losses) == len(epochs_x):
        plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    if len(logits_images) == len(epochs_x):
        plt.plot(epochs_x, logits_images, label='Average Logits')
    if len(logits_texts) == len(epochs_x):
        plt.plot(epochs_x, logits_texts, label='Average Logits')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, processor, max_length=77):
        self.image_folder = image_folder
        self.processor = processor
        self.max_length = max_length
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')

        labels = self.annotations[self.image_paths[idx]]
        
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]
        else:
            label = ''

        inputs = self.processor(images=image, text=label, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return inputs['pixel_values'].squeeze(0).half(), inputs['input_ids'].squeeze(0)  # Convert image to half precision

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate loss as the mean of the two cross-entropy losses
        loss_img = self.criterion(logits, labels)
        loss_txt = self.criterion(logits.t(), labels)

        return (loss_img + loss_txt) / 2

# Custom hook to scale the feature activation
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_indices, scale_factor):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_indices = feature_indices
        self.scale_factor = scale_factor
        self.handle = None
        self.register_hook()

    def hook_fn(self, module, input, output):
        for feature_idx in self.feature_indices:
            output[:, :, feature_idx] *= self.scale_factor
        return output

    def register_hook(self):
        layer = self.model.vision_model.encoder.layers[self.layer_idx].mlp.fc1
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

def register_hooks(model, modified_neurons_layers, scale_factors):
    hooks = []
    for layer_idx, feature_indices in modified_neurons_layers.items():
        scale_factor = scale_factors[layer_idx]
        hook = FeatureScalerHook(model, layer_idx, feature_indices, scale_factor)
        hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
        
        
# Define the neurons to tamper with, and scaling factors for each layer
# Penultimate layer 22, Feature 2432 is an "adverb neuron".
# When scaled to x1000, CLIP will predict mainly adverbs for any image.
# See https://github.com/zer0int/Golden-Gate-CLIP for details
modified_neurons_layers = {
    31: [281],
    28: [168, 1297],
    30: [2432]
}
scale_factors = {
    31: 100,
    28: 100,
    30: 1000
}

contrastive_loss = ContrastiveLoss(temperature=0.07)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

clipmodel = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(clipmodel).to(device).half()  # Convert model to half precision
processor = CLIPProcessor.from_pretrained(clipmodel)

def preprocess(images, texts):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return inputs['pixel_values'], inputs['input_ids']

#For continuing training a model checkpoint
#_, preprocess = clip.load(clipmodel, device=device)
#model = torch.load("continue/training/my/finetune.pt")
#model = model.cuda()

unfreeze_all = True

# Training configuration
# This configuraation is tested on 1 x Nvidia A100. I don't know will this setting will winning the CLIP heart :v
EPOCHS = 20
batch_size = 20
max_learning_rate = 1e-7
learning_rate = 5e-8

# Define your training dataset and dataloader, or use below to reproduce results
max_length = 77  # CLIP's default max length
dataset1 = ImageTextDataset("G:\SomniumSC2-datasets", "G:\SomniumSC2-datasets\my-dataset-labels.json", processor, max_length=max_length)
concatenated_dataset = ConcatDataset([dataset1])
train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)
val_dataset = ImageTextDataset("G:\Validation", "G:\Validation\my-dataset-labels.json", processor, max_length=max_length)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
total_steps = len(train_dataloader) * EPOCHS

# Define parameter groups for different learning rates
visual_parameters = [p for n, p in model.named_parameters() if 'visual' in n and p.requires_grad]
text_parameters = [p for n, p in model.named_parameters() if 'text' in n and p.requires_grad]
other_parameters = [p for n, p in model.named_parameters() if 'visual' not in n and 'text' not in n and p.requires_grad]

param_groups = [
    {'params': visual_parameters, 'lr': 1e-7},
    {'params': text_parameters, 'lr': 5e-8},
    {'params': other_parameters, 'lr': 1e-7},
]

accumulation_steps = 4  # Effective batch size will be batch_size * accumulation_steps

optimizer = AdaBelief([p for p in model.parameters() if p.requires_grad], lr=learning_rate, eps=1e-14, betas=(0.9, 0.999), weight_decay=1e-3, weight_decouple=True, rectify=True, print_change_log=False)

scheduler = OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')

model = model.float()

print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print("== START == \n")

def trainloop():
    contrastive_loss = ContrastiveLoss(temperature=0.07).to(device)
    logits_images = []
    logits_texts = []

    accumulation_steps = 2  # Adjust as needed to simulate larger batch size
    scaler = GradScaler()
    # Register hooks to tamper with activation value
    hooks = register_hooks(model, modified_neurons_layers, scale_factors)
    for epoch in range(EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
        train_dataloader_prog = train_dataloader
        train_dataloader_all = train_dataloader
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)

        optimizer.zero_grad()  # Reset gradients at the beginning of the epoch

        for batch_idx, (images, texts) in progress_bar:
            images, texts = images.to(device), texts.to(device)
            batch_logits_images = []
            batch_logits_texts = []

            with autocast():
                outputs = model(pixel_values=images, input_ids=texts)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                total_loss = contrastive_loss(logits_per_image, logits_per_text)
                acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            batch_logits_images.append(logits_per_image.mean().item())
            batch_logits_texts.append(logits_per_text.mean().item())

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_norm = parameter.grad.norm().item()
                    gradient_norms.setdefault(name, []).append(grad_norm)

            monitor_gradient_norms(gradient_norms)

            total_train_loss += total_loss.item()

            progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}  --  Logits Image: {batch_logits_images[-1]:.3f}, Text: {batch_logits_texts[-1]:.3f}'})

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)

        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)

        plot_gradient_norms(gradient_norms, epoch)

        epoch_train_acc = sum(train_accs) / len(train_accs)
        epoch_train_f1 = sum(train_f1s) / len(train_f1s)
        with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
            f.write(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_train_loss:.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")

        model.eval()
        total_val_loss = 0.0
        print("Running Validation...")
        with torch.no_grad():
            for images, texts in val_dataloader:
                images, texts = images.to(device), texts.to(device)
                outputs = model(pixel_values=images, input_ids=texts)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                val_loss = contrastive_loss(logits_per_image, logits_per_text)
                total_val_loss += val_loss.item()
                val_acc, val_f1 = calculate_metrics(logits_per_image, ground_truth)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

        avg_val_loss = total_val_loss / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        if epoch >= 1:
            plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)

        epoch_val_acc = sum(val_accs) / len(val_accs)
        epoch_val_f1 = sum(val_f1s) / len(val_f1s)

        if epoch >= 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
            plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.savefig(f"{plots_folder}/loss_plot_epoch_{epoch + 1}.png")
            plt.close()

        print(Fore.YELLOW + "======================== STATS =============================")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)

        with open(f"{text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
            f.write("======================== STATS =============================\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            f.write("============================================================\n")

        if (epoch + 1) % 1 == 0 or epoch == EPOCHS - 1:
            model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
            remove_hooks(hooks)# Remove hooks
            torch.save(model, model_path)
            print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)
            hooks = register_hooks(model, modified_neurons_layers, scale_factors)# Re-attach hooks

    remove_hooks(hooks)# After training

trainloop()
