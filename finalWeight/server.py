import os
import flwr as fl
import numpy as np
import json
from config import SERVER_ADDRESS, NUM_CLIENTS

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        num_rounds = kwargs.pop('num_rounds', 1)
        super().__init__(**kwargs)
        self.metrics_history = []
        self.train_metrics_history = []
        self.best_metrics = {
            'mse': (float('inf'), -1),
            'rmse': (float('inf'), -1),
            'mae': (float('inf'), -1),
            'r2': (float('-inf'), -1)
        }
        self.num_rounds = num_rounds
        self.best_weights = None  # Store the best weights
        self.aggregated_weights = None  # Initialize aggregated_weights

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)

        num_examples_list = [fit_res.num_examples for _, fit_res in results]
        train_mse_values = [fit_res.metrics['train_mse'] for _, fit_res in results]
        train_rmse_values = [fit_res.metrics['train_rmse'] for _, fit_res in results]
        train_mae_values = [fit_res.metrics['train_mae'] for _, fit_res in results]

        avg_train_mse = np.average(train_mse_values, weights=num_examples_list)
        avg_train_rmse = np.average(train_rmse_values, weights=num_examples_list)
        avg_train_mae = np.average(train_mae_values, weights=num_examples_list)

        self.train_metrics_history.append({
            'round': rnd,
            'train_mse': avg_train_mse,
            'train_rmse': avg_train_rmse,
            'train_mae': avg_train_mae
        })

        print(f"Aggregated training metrics for round: {rnd}")
        print(f"Train MSE: {avg_train_mse:.4f}, Train RMSE: {avg_train_rmse:.4f}, Train MAE: {avg_train_mae:.4f}")

        # Store aggregated weights to be accessed in aggregate_evaluate
        self.aggregated_weights = aggregated_weights

        return aggregated_weights, {}

    def aggregate_evaluate(self, rnd, results, failures):
        if failures:
            print(f"Round {rnd} encountered {len(failures)} failures.")

        num_examples_list = [r.num_examples for _, r in results]
        mse_values = [r.metrics['mse'] for _, r in results]
        rmse_values = [r.metrics['rmse'] for _, r in results]
        mae_values = [r.metrics['mae'] for _, r in results]
        r2_values = [r.metrics['r2'] for _, r in results]

        avg_mse = np.average(mse_values, weights=num_examples_list)
        avg_rmse = np.average(rmse_values, weights=num_examples_list)
        avg_mae = np.average(mae_values, weights=num_examples_list)
        avg_r2 = np.average(r2_values, weights=num_examples_list)

        self.metrics_history.append({
            'round': rnd,
            'mse': avg_mse,
            'rmse': avg_rmse,
            'mae': avg_mae,
            'r2': avg_r2
        })

        if avg_mse < self.best_metrics['mse'][0]:
            self.best_metrics['mse'] = (avg_mse, rnd)
            self.best_weights = self.aggregated_weights  # Update best weights
            self.save_best_weights(self.aggregated_weights)  # Save the best weights
        if avg_rmse < self.best_metrics['rmse'][0]:
            self.best_metrics['rmse'] = (avg_rmse, rnd)
        if avg_mae < self.best_metrics['mae'][0]:
            self.best_metrics['mae'] = (avg_mae, rnd)
        if avg_r2 > self.best_metrics['r2'][0]:
            self.best_metrics['r2'] = (avg_r2, rnd)

        print(f"Round {rnd} evaluation metrics:")
        print(f"MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}")

        if rnd == self.num_rounds:
            final_metrics = {
                'mse': [],
                'rmse': [],
                'mae': [],
                'r2': []
            }

            for metrics in self.metrics_history:
                final_metrics['mse'].append((metrics['mse'], metrics['round']))
                final_metrics['rmse'].append((metrics['rmse'], metrics['round']))
                final_metrics['mae'].append((metrics['mae'], metrics['round']))
                final_metrics['r2'].append((metrics['r2'], metrics['round']))

            print("Final evaluation summary:")
            for metric, values in final_metrics.items():
                best_value, best_round = self.best_metrics[metric]
                print(f"Best {metric.upper()}: {best_value:.4f} (Round {best_round})")

            with open("metrics_history.json", "w") as f:
                json.dump({
                    'train_metrics': self.train_metrics_history,
                    'eval_metrics': self.metrics_history
                }, f, indent=4)

        return avg_mse, {}

    def save_best_weights(self, weights):
        np.save("best_weights.npy", weights)  # Save the best weights to a file

    # def load_best_weights(self):
    #     if os.path.exists("/app/best_weights.npy"):
    #         self.best_weights = np.load("/app/best_weights.npy", allow_pickle=True)

def main():
    num_rounds = 10
    strategy = CustomStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
        on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
        num_rounds=num_rounds
    )
    # strategy.load_best_weights()  # Load best weights if they exist
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
    )

if __name__ == "__main__":
    main()
