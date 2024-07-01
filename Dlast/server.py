# import flwr as fl
# import numpy as np
# from config import SERVER_ADDRESS, NUM_CLIENTS

# class CustomStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(self, rnd, results, failures):
#         aggregated_weights = super().aggregate_fit(rnd, results, failures)
#         return aggregated_weights

#     def aggregate_evaluate(self, rnd, results, failures):
#         losses = [r.metrics['loss'] for _, r in results]
#         val_losses = [r.metrics['val_loss'] for _, r in results if 'val_loss' in r.metrics]
#         num_samples = [r.num_examples for _, r in results]
#         total_samples = sum(num_samples)
#         if total_samples == 0:
#             print(f"Round {rnd} aggregated evaluation received no samples")
#             return 0, {}
#         weighted_mse = sum(mse * n for mse, n in zip(losses, num_samples)) / total_samples
#         weighted_val_mse = sum(val_mse * n for val_mse, n in zip(val_losses, num_samples)) / total_samples
#         weighted_rmse = np.sqrt(weighted_mse)
#         weighted_val_rmse = np.sqrt(weighted_val_mse)
#         print(f"Round {rnd} aggregated evaluation MSE: {weighted_mse}")
#         print(f"Round {rnd} aggregated evaluation RMSE: {weighted_rmse}")
#         return weighted_mse, {"val_loss": weighted_val_mse}

# def fit_metrics_aggregation_fn(results):
#     losses = [r.metrics['loss'] for _, r in results]
#     val_losses = [r.metrics['val_loss'] for _, r in results if 'val_loss' in r.metrics]
#     num_samples = [r.num_examples for _, r in results]
#     total_samples = sum(num_samples)
#     weighted_loss = sum(loss * n for loss, n in zip(losses, num_samples)) / total_samples
#     weighted_val_loss = sum(val_loss * n for val_loss, n in zip(val_losses, num_samples)) / total_samples
#     return {"loss": weighted_loss, "val_loss": weighted_val_loss}

# def main():
#     strategy = CustomStrategy(
#         min_fit_clients=NUM_CLIENTS,
#         min_eval_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn
#     )
#     fl.server.start_server(
#         server_address=SERVER_ADDRESS,
#         config=fl.server.ServerConfig(num_rounds=1),
#         grpc_max_message_length=1024*1024*1024,
#         strategy=strategy
#     )

# if __name__ == "__main__":
#     main()
import flwr as fl
import numpy as np
from config import SERVER_ADDRESS, NUM_CLIENTS

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        losses = [r.metrics['loss'] for _, r in results]
        val_losses = [r.metrics['val_loss'] for _, r in results if 'val_loss' in r.metrics]
        num_samples = [r.num_examples for _, r in results]
        total_samples = sum(num_samples)
        if total_samples == 0:
            print(f"Round {rnd} aggregated evaluation received no samples")
            return 0, {}
        weighted_mse = sum(mse * n for mse, n in zip(losses, num_samples)) / total_samples
        weighted_val_mse = sum(val_mse * n for val_mse, n in zip(val_losses, num_samples)) / total_samples
        weighted_rmse = np.sqrt(weighted_mse)
        weighted_val_rmse = np.sqrt(weighted_val_mse)
        print(f"Round {rnd} aggregated evaluation MSE: {weighted_mse}")
        print(f"Round {rnd} aggregated evaluation RMSE: {weighted_rmse}")
        return weighted_mse, {"val_loss": weighted_val_mse}

def fit_metrics_aggregation_fn(results):
    losses = [r.metrics['loss'] for _, r in results]
    val_losses = [r.metrics['val_loss'] for _, r in results if 'val_loss' in r.metrics]
    num_samples = [r.num_examples for _, r in results]
    total_samples = sum(num_samples)
    weighted_loss = sum(loss * n for loss, n in zip(losses, num_samples)) / total_samples
    weighted_val_loss = sum(val_loss * n for val_loss, n in zip(val_losses, num_samples)) / total_samples
    return {"loss": weighted_loss, "val_loss": weighted_val_loss}

def main():
    strategy = CustomStrategy(
        fraction_fit=1.0,  # Sample all clients for training
        fraction_evaluate=1.0,  # Sample all clients for evaluation
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
        on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn
    )
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=1),
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
    )

if __name__ == "__main__":
    main()
