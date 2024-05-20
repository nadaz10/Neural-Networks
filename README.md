# Digit Recognition Model Training

## Overview

This project demonstrates the process of training a neural network for digit recognition using the MNIST dataset. The model is built with PyTorch and undergoes various training experiments with different learning rates and batch sizes to find the optimal hyperparameters.

## Setup

### Requirements

- Python 3.6 or higher
- PyTorch
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/digit-recognition.git
    cd digit-recognition
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

1. Load the dataset:
    ```python
    df = pd.read_csv('train.csv')
    ```
2. Split the dataset into features and labels:
    ```python
    X = df.drop('label', axis=1)  # Features
    y = df['label']  # Labels
    ```
3. Split the dataset into training and validation sets:
    ```python
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
4. Normalize the data:
    ```python
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    df_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    df_val_normalized = pd.DataFrame(X_val_normalized, columns=X_val.columns)
    ```

## Custom Dataset and DataLoader

1. Define a custom PyTorch dataset class:
    ```python
    class CustomDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features.values, dtype=torch.float32)
            self.labels = torch.tensor(labels.values, dtype=torch.long)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    ```
2. Create datasets and data loaders:
    ```python
    train_dataset = CustomDataset(df_train_normalized, y_train)
    val_dataset = CustomDataset(df_val_normalized, y_val)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ```

## Model Architecture

1. Define the neural network model:
    ```python
    class DigitRecModel(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(DigitRecModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return x
    ```

## Training the Model

1. Initialize the model, loss function, and optimizer:
    ```python
    input_size = 28 * 28  # Assuming images are 28x28 pixels
    output_size = 10  # Number of classes (digits 0-9)
    hidden_size1 = 256  
    hidden_size2 = 128
    simple_model = DigitRecModel(input_size, hidden_size1, hidden_size2, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
    ```

2. Train and validate the model:
    ```python
    epochs = 3
    for epoch in range(epochs):
        simple_model.train()
        correct_train = 0
        total_train = 0
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = simple_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_train_loss += loss.item()
        train_accuracy = correct_train / total_train
        train_losses.append(running_train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        simple_model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = simple_model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                running_val_loss += loss.item()
        val_accuracy = correct_val / total_val
        val_losses.append(running_val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
    ```

3. Experiment with different learning rates and batch sizes:
    ```python
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    batch_sizes = [32, 64, 128, 256, 512]
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Model, optimizer, and loaders initialization
            # Training and validation loops as above
    ```

## Results Visualization

1. Plotting Loss and Accuracy:
    ```python
    plt.figure(figsize=(18, 9))

    plt.subplot(2, 2, 1)  # Train Loss plot
    for i, lr in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            label = f'LR: {lr}, BS: {batch_size}'
            start_idx = (i * len(batch_sizes) + j) * epochs
            end_idx = start_idx + epochs
            plt.plot(range(1, epochs + 1), train_losses[start_idx:end_idx], label=label)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 2)  # Validation Loss plot
    for i, lr in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            label = f'LR: {lr}, BS: {batch_size}'
            start_idx = (i * len(batch_sizes) + j) * epochs
            end_idx = start_idx + epochs
            plt.plot(range(1, epochs + 1), val_losses[start_idx:end_idx], label=label)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 3)  # Training Accuracy plot
    for i, lr in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            label = f'LR: {lr}, BS: {batch_size}'
            start_idx = (i * len(batch_sizes) + j) * epochs
            end_idx = start_idx + epochs
            plt.plot(range(1, epochs + 1), train_accuracies[start_idx:end_idx], label=label)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)  # Validation Accuracy plot
    for i, lr in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            label = f'LR: {lr}, BS: {batch_size}'
            start_idx = (i * len(batch_sizes) + j) * epochs
            end_idx = start_idx + epochs
            plt.plot(range(1, epochs + 1), val_accuracies[start_idx:end_idx], label=label)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    ```
