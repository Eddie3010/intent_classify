import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- 1. Load Data ---
data_path = "ai_sales_intent_dataset_1000_samples.csv"
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# Handle potential missing values
df.fillna('', inplace=True)

print("Data loaded. Number of rows:", len(df))

# --- 2. Preprocessing ---
print("Initializing Preprocessors...")

# Target encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['current_intent'])
num_classes = len(label_encoder.classes_)

# Feature 1: Categorical 'previous_intent'
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_prev_intent = ohe.fit_transform(df[['previous_intent']])

# Feature 2 & 3: Text
tfidf_bot = TfidfVectorizer(max_features=500)
X_bot_resp = tfidf_bot.fit_transform(df['previous_bot_response']).toarray()

tfidf_query = TfidfVectorizer(max_features=500)
X_query = tfidf_query.fit_transform(df['current_query']).toarray()

# Combine all features
X = np.hstack((X_prev_intent, X_bot_resp, X_query))
print("Combined Feature shape:", X.shape)

input_size = X.shape[1]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Split into train/validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_encoded)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 3. Model Definition ---
class IntentClassifierMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IntentClassifierMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = IntentClassifierMLP(input_size, num_classes)
print(model)

# --- 4. Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

train_losses = []
val_losses = []

print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_acc = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# --- 5. Evaluation and Plots ---
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Plot Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.close()
print("\nSaved loss_curve.png")

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix on Validation Set')
plt.xlabel('Predicted Intent')
plt.ylabel('True Intent')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Saved confusion_matrix.png")

# Classification Report (similar to Correlations report)
cr = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
print("\nClassification Report:\n", cr)

# Plot Classification Report as Heatmap
report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1] # exclude support and averages
plt.figure(figsize=(8, 6))
sns.heatmap(report_df, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Classification Metrics Correlation / Report Summary')
plt.tight_layout()
plt.savefig('classification_metrics.png')
plt.close()
print("Saved classification_metrics.png")

# --- 6. Save Artifacts ---
print("\nSaving Models and Preprocessors...")
torch.save(model.state_dict(), "model.pth")

preprocessors = {
    'label_encoder': label_encoder,
    'ohe': ohe,
    'tfidf_bot': tfidf_bot,
    'tfidf_query': tfidf_query,
    'input_size': input_size,
    'num_classes': num_classes
}
joblib.dump(preprocessors, "preprocessors.pkl")

print("Training finished successfully. Saved model.pth and preprocessors.pkl.")
