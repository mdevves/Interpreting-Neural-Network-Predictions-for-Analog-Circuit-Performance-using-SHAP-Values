import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import shap


# Load dataset
file_path = "data_for_NN.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

df = pd.read_csv(file_path)

# Extract features (5 inputs) and performance metrics (8 outputs)
scaler_X =  StandardScaler()
scaler_y =  StandardScaler()


X = scaler_X.fit_transform(df.iloc[:, :5].values)  # Normalize inputs
y = scaler_y.fit_transform(df.iloc[:, 5:].values)  # Normalize outputs



# === ADD: Πίνακες ορίων κανονικοποίησης (inputs & outputs) ===
# Ονόματα στηλών
X_names = df.columns[:5].tolist()
y_names = df.columns[5:].tolist()

# Στατιστικά από τους ήδη fitted scalers
mu_X  = scaler_X.mean_          # (5,)
sig_X = scaler_X.scale_         # (5,)
mu_y  = scaler_y.mean_          # (8,)
sig_y = scaler_y.scale_         # (8,)

# Ελάχιστα/Μέγιστα στο αρχικό dataset (σε φυσικές μονάδες)
X_min = df.iloc[:, :5].min().values
X_max = df.iloc[:, :5].max().values
y_min = df.iloc[:, 5:].min().values
y_max = df.iloc[:, 5:].max().values

# Τα αντίστοιχα “όρια” στον z-space (κανονικοποιημένος χώρος)
# z = (x - μ) / σ
X_min_z = (X_min - mu_X) / sig_X
X_max_z = (X_max - mu_X) / sig_X
y_min_z = (y_min - mu_y) / sig_y
y_max_z = (y_max - mu_y) / sig_y

# Συνάρτηση που φτιάχνει LaTeX πίνακα
def latex_limits_table(names, mu, sig, vmin, vmax, zmin, zmax, title):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Όρια κανονικοποίησης ({title}). Για κάθε παράμετρο: μέση τιμή $\mu$, τυπική απόκλιση $\sigma$, ελάχιστο/μέγιστο στο δείγμα και τα αντίστοιχα όρια στον κανονικοποιημένο χώρο $z=\frac{{x-\mu}}{{\sigma}}$.}}"
    )
    lines.append(r"\begin{tabular}{|l|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Παράμετρος} & $\mu$ (orig) & $\sigma$ (orig) & $\min$ (orig) & $\max$ (orig) & $\min$ (z) & $\max$ (z) \\ \hline")
    for i, n in enumerate(names):
        lines.append(
            rf"\textlatin{{{n}}} & {mu[i]:.6g} & {sig[i]:.6g} & {vmin[i]:.6g} & {vmax[i]:.6g} & {zmin[i]:.3g} & {zmax[i]:.3g} \\ \hline"
        )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# Δημιουργία LaTeX πινάκων
latex_inputs  = latex_limits_table(X_names, mu_X, sig_X, X_min, X_max, X_min_z, X_max_z, "Είσοδοι")
latex_outputs = latex_limits_table(y_names, mu_y, sig_y, y_min, y_max, y_min_z, y_max_z, "Έξοδοι")

# Εκτύπωση για copy-paste στο κείμενο
print("\n\n=== LaTeX (Inputs) ===\n")
print(latex_inputs)
print("\n\n=== LaTeX (Outputs) ===\n")
print(latex_outputs)

# Προαιρετικά: αποθήκευση σε αρχεία .tex για \input{} στο LaTeX
with open("limits_inputs.tex", "w", encoding="utf-8") as f:
    f.write(latex_inputs)
with open("limits_outputs.tex", "w", encoding="utf-8") as f:
    f.write(latex_outputs)





# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define Neural Network
class PerformanceNet(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=8):
        super(PerformanceNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = PerformanceNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 3000
train_losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot καμπύλης σύγκλισης
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Καμπύλη σύγκλισης της εκπαίδευσης")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X_mean': scaler_X.mean_,
    'scaler_X_scale': scaler_X.scale_,
    'scaler_y_mean': scaler_y.mean_,
    'scaler_y_scale': scaler_y.scale_
}, "performance_model.pth")
print("Model and scalers saved as performance_model.pth")

# Evaluate on test data
model.eval()
test_predictions = model(X_test)
test_loss = criterion(test_predictions, y_test)
print(f"Test Loss: {test_loss.item():.4f}")


mse = test_loss.item()
rmse = np.sqrt(mse)
r2 = r2_score(y_test.detach().numpy(), test_predictions.detach().numpy())

# Display results
print(f"Test Loss (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")



# SHAP ανά έξοδο (1–8) με πραγματικά ονόματα εξόδων και αποθήκευση plot με ασφαλές filename
model.eval()
background = X_train[:50].numpy()
test_samples = X_test[:100].numpy()
feature_names = df.columns[:5]

for output_index in range(8):
    out_name = y_names[output_index]                       # πραγματικό όνομα εξόδου (από CSV)
    safe_name = (str(out_name)
                 .replace("/", "_")
                 .replace("\\", "_")
                 .replace(":", "_")
                 .replace("*", "_")
                 .replace("?", "")
                 .replace("\"", "")
                 .replace("<", "_")
                 .replace(">", "_")
                 .replace("|", "_")
                 .replace(" ", "_"))                       # ασφαλές για όνομα αρχείου

    print(f"\n➡ Υπολογισμός SHAP για έξοδο {out_name}")

    # Προβλεπτική συνάρτηση για συγκεκριμένη έξοδο
    def predict_output_i(x_numpy, idx=output_index):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            return model(x_tensor).detach().numpy()[:, idx:idx+1]

    # KernelExplainer για αυτή την έξοδο
    explainer = shap.KernelExplainer(predict_output_i, background)

    # SHAP values για τα test samples
    shap_values = explainer.shap_values(test_samples)

    # Έλεγχος και plot
    values = np.array(shap_values).squeeze()
    if values.ndim != 2 or values.shape[1] != test_samples.shape[1]:
        print(f"❌ Παράλειψη {out_name} λόγω λάθους shape: {values.shape}")
        continue
    if np.isnan(values).any():
        print(f"❌ Παράλειψη {out_name} λόγω NaN")
        continue

    print(f"✅ SHAP Summary Plot για {out_name}")
    plt.figure()
    shap.summary_plot(values, test_samples, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary for {out_name}", fontsize=14)
    plt.tight_layout()
    # αποθήκευση με πραγματικό όνομα εξόδου (ασφαλές filename)
    plt.savefig(f"Shap summary plot for {safe_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Υπολογισμός μέσου όρου και τυπικής απόκλισης ανά χαρακτηριστικό (|SHAP|)
    means_abs = np.abs(values).mean(axis=0)
    stds_abs  = np.abs(values).std(axis=0)

    print("Feature\t\tMean |SHAP|\tStd |SHAP|")
    print("-" * 40)
    for i, fname in enumerate(feature_names):
        print(f"{fname:<10s}\t{means_abs[i]: .4f}\t\t{stds_abs[i]: .4f}")



# Μετατροπή των δεδομένων σε NumPy arrays για plotting
y_true = y_test.detach().numpy()
y_pred = test_predictions.detach().numpy()

# Δημιουργία scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions vs Actual")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Ideal Fit (y = x)")

# Προσθήκη labels και τίτλου
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Comparison of Actual and Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

# Example: Make a prediction on new data
X_sample = torch.tensor([[16.0, 7.0, 0.6, 6.0, 33.0]], dtype=torch.float32)
predictions = model(X_sample)
print("Predicted Performance Metrics:", predictions.detach().numpy())

