# Workload Prediction - CPU Trace Federated Learning

This project implements a federated learning approach for workload prediction using CPU traces.

## Prerequisites

- Python 3.x
- Google Cloud SDK
- SSH key pair

## Dependencies

Install the required Python packages:

```
tensorflow==2.16.1
flwr==1.8.0
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
pykalman==0.9.7
```

You can install these dependencies using:

```
pip install -r requirements.txt
```

## Google Cloud Setup

1. Install and initialize Google Cloud SDK:
   ```
   gcloud init
   ```

2. Generate SSH keys (if needed):
   ```
   ssh-keygen -t rsa -b 2048 -C "your_email@example.com"
   ```

3. List your VM instances:
   ```
   gcloud compute instances list
   ```

4. Connect to your VM instance:
   ```
   gcloud compute ssh [YOUR_VM_NAME] --zone [YOUR_VM_ZONE]
   ```

## Project Structure

The main code is located in the `stage2` folder.

## Uploading Code to VM

Navigate to your project directory on your local machine:

```
cd path/to/your/project
```

Upload your code to the VM:

```
gcloud compute scp --recurse * node1instance-20240614-035151:~/ --zone us-central1-b
```

## Running the Project

1. First, run the file distribution script:
   ```
   python distribute_files.py
   ```

2. In one terminal, start the server:
   ```
   python start_server.py
   ```

3. In another terminal, start the clients:
   ```
   python start.py
   ```

## Results and Visualization

- The results will be saved in `metrics_history.json`.
- To visualize the results, run:
  ```
  python plotting.py
  ```

## Files

- `client.py`: Implements the federated learning client.
- `config.py`: Contains configuration settings.
- `model.py`: Defines the machine learning model.
- `requirements.txt`: Lists the project dependencies.
- `distribute_files.py`: Distributes files to clients.
- `start_server.py`: Starts the federated learning server.
- `start.py`: Initiates the federated learning process.
- `plotting.py`: Generates visualizations from the results.

## Note

Ensure all necessary permissions are set up on your Google Cloud VM and that you have the required access to run these scripts.
