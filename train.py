import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from nn import ResNet

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(batch_size=64, val_split=0.1):
    """
    Load and preprocess the MNIST dataset.
    
    Args:
        batch_size: Batch size for training and validation
        val_split: Proportion of training data to use for validation
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Split training data into training and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load test data
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, epochs=15, lr=0.001):
    """
    Train the model.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        history: Dictionary containing training and validation metrics
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize history dictionary to store metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize best validation accuracy for model checkpointing
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        end_time = time.time()
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Time: {end_time-start_time:.2f}s - '
              f'Train Loss: {train_loss:.4f} - '
              f'Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f} - '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save model if it has the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
    
    return history

def evaluate(model, test_loader):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        
    Returns:
        test_acc: Test accuracy
        confusion_mat: Confusion matrix
    """
    model.eval()
    test_correct = 0
    test_total = 0
    confusion_mat = torch.zeros(10, 10)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output.data, 1)
            
            # Update statistics
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            # Update confusion matrix
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_mat[t.long(), p.long()] += 1
    
    # Calculate test accuracy
    test_acc = 100.0 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return test_acc, confusion_mat

def plot_metrics(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training and validation metrics
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def plot_confusion_matrix(confusion_mat):
    """
    Plot confusion matrix.
    
    Args:
        confusion_mat: Confusion matrix
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = [str(i) for i in range(10)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, int(confusion_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    # Hyperparameters
    batch_size = 64
    epochs = 15
    learning_rate = 0.001
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = ResNet().to(device)
    print(model)
    
    # Train model
    print("Training model...")
    history = train(model, train_loader, val_loader, epochs=epochs, lr=learning_rate)
    
    # Load best model
    print("Loading best model...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate model
    print("Evaluating model...")
    test_acc, confusion_mat = evaluate(model, test_loader)
    
    # Plot metrics
    plot_metrics(history)
    plot_confusion_matrix(confusion_mat)
    
    print(f"Final test accuracy: {test_acc:.2f}%")
    if test_acc >= 99.0:
        print("Success! Achieved 99%+ accuracy on MNIST.")
    else:
        print("Target accuracy of 99%+ not reached. Consider adjusting hyperparameters or model architecture.")

if __name__ == "__main__":
    main()
