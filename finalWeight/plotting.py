import json
import matplotlib.pyplot as plt

# Load metrics from JSON file
with open("metrics_history.json", "r") as f:
    metrics = json.load(f)

train_metrics = metrics['train_metrics']
eval_metrics = metrics['eval_metrics']

rounds = [m['round'] for m in train_metrics]

# Extract metrics
train_mse = [m['train_mse'] for m in train_metrics]
train_rmse = [m['train_rmse'] for m in train_metrics]
train_mae = [m['train_mae'] for m in train_metrics]

eval_mse = [m['mse'] for m in eval_metrics]
eval_rmse = [m['rmse'] for m in eval_metrics]
eval_mae = [m['mae'] for m in eval_metrics]
eval_r2 = [m['r2'] for m in eval_metrics]

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(rounds, train_mse, label='Train MSE')
plt.plot(rounds, eval_mse, label='Eval MSE')
plt.xlabel('Rounds')
plt.ylabel('MSE')
plt.title('Train vs Eval MSE')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(rounds, train_rmse, label='Train RMSE')
plt.plot(rounds, eval_rmse, label='Eval RMSE')
plt.xlabel('Rounds')
plt.ylabel('RMSE')
plt.title('Train vs Eval RMSE')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(rounds, train_mae, label='Train MAE')
plt.plot(rounds, eval_mae, label='Eval MAE')
plt.xlabel('Rounds')
plt.ylabel('MAE')
plt.title('Train vs Eval MAE')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(rounds, eval_r2, label='Eval R2')
plt.xlabel('Rounds')
plt.ylabel('R2')
plt.title('Eval R2')
plt.legend()

plt.tight_layout()
plt.show()
