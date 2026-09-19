import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
import numpy as np
import pandas as pd

count = 0

vocab_size = 100  # Total number of input tokens
embedding_dim = 10000
hidden_dim = 128
max_seq_len = 1000
learning_rate = 0.00001
num_epochs = 20
batch_size = 1
bidirectional = True  # for LSTM
model_save_path = "best_consolidation_model10000_0.00001_20.pth"    
# Device configuration
device = "cuda:0"
print(f"Using device: {device}")


class PredictionConsolidatorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len, bidirectional=True):
        super(PredictionConsolidatorLSTM, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        # Input size is now embedding dim + 1 for the confidence value
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True, bidirectional=bidirectional)
        # Output layer to 100 classes (99 valid jerseys, -1 for no prediction)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 100)  # Output a single number
        print("++++++++++++++++++++++++++++++++++++++++1111111++++++++++++++++++++++++++++++++++++++")

    def forward(self, x):
        x[:, 0, 0] = x[:, 0, 0].clone().masked_fill_(x[:, 0, 0] > 99.0, 0.0)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(x.shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("x" ,x)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("x" ,x[:,0,0])
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("x" ,x[:,0,0].long())
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        embedded = self.embedding(x[:,0,0].long())  # Shape (batch_size, max_seq_len, embedding_dim) - Extract first feature for embedding (jersey number tokens)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("x[:, 0]",x[:, 0].shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("embedded",embedded.shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("x[:, 1]", x[:, 1].shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        concatenated_input = torch.cat((embedded, x[:,0,  1].unsqueeze(1)), dim=1)  # Shape (batch_size, max_seq_len, embedding_dim+1) - adding confidence as a feature
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("concatenated_input", concatenated_input.shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")        
        lstm_out, _ = self.lstm(concatenated_input)  # Shape (batch_size, max_seq_len, hidden_dim * (2 if bidirectional else 1))
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("lstm_out", lstm_out.shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")          
        # Average pooling to get single representation
        pooled_output = torch.mean(lstm_out, dim=0)  # Shape (batch_size, hidden_dim * (2 if bidirectional else 1))
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("pooled_output", pooled_output.shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") 
        output = self.fc(pooled_output)  # shape: (batch_size, 100)
        output = torch.softmax(output, dim=0) #Apply softmax to the outputs
        return output

    def get_prediction(self, output):
        # Get class prediction from the output of the model
        predicted_class = torch.argmax(output, dim = 0) # get the most probable class
        predicted_number = predicted_class # The class index is our predicted number.
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("predicted_number", predicted_number)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return predicted_number


def is_valid_number(value):
    """Check if a value is a valid number (integer or string representation)"""
    if isinstance(value, int):
        return True
    if isinstance(value, str) and value.isdigit():
        return True
    return False

def calculate_accuracy(predictions, labels):
    """Calculate the accuracy from predictions and labels"""
    predictions = torch.tensor(predictions).to('cuda:0')
    labels = torch.tensor(labels).to('cuda:0')
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("predicted_labels", predictions)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("labels", labels)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("accuracy", accuracy)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return accuracy

def find_best_prediction_with_lstm(results, model, device, max_seq_len, vocab_size):
    """Predict the best jersey number using the LSTM model."""
    if not results:
        return -1, [], []  # Return defaults if no results
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("results", results)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Extract the jersey numbers and confidences
    jersey_numbers = [int(item[0]) for item in results]
    confidences = [float(item[1]) for item in results]

    # Create the input sequence for the LSTM with jersey numbers and confidence values
    sequence = []
    for number, confidence in results:
       sequence.append([int(number), float(confidence)]) # convert to the correct type and apply offset to jersey numbers

    padded_sequence = sequence + [[0, 0]] * (max_seq_len - len(sequence))
    sequence_tensor = torch.tensor(padded_sequence, dtype=torch.float).unsqueeze(1).to(device)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("padded_sequence", len(padded_sequence))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("sequence", len(sequence))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("sequence_tensor", sequence_tensor)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    model.eval()
    with torch.no_grad():
    #    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #    print("output", output)
    #    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
       output = model(sequence_tensor)  # shape: (1, 100) - batch size of 1
    #    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #    print("output", output)
    #    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


       prediction = model.get_prediction(output)
    #    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #    print("prediction", prediction.item())
    #    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
       predicted_number = prediction.item()
    all_unique = list(set(jersey_numbers))
    # calculate weights as product of probabilities
    weights = [1] * len(confidences)
    for i, conf in enumerate(confidences):
      weights[i] = weights[i]* conf
    # count = count + 1
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("count", count)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return predicted_number, all_unique, weights

def process_jersey_id_predictions_Internal(file_path, model, device, max_seq_len, vocab_size, useBias=False):
    all_results = {}
    final_results = {}
    with open(file_path, 'r') as f:
        results_dict = json.load(f)
    for name in results_dict.keys():
        tmp = name.split('_')
        tracklet = tmp[0]

        if tracklet not in all_results:
            all_results[tracklet] = []
            final_results[tracklet] = -1  # default
        value = results_dict[name]['label']
        if not is_valid_number(value):
            continue
        confidence = results_dict[name]['confidence']
        # ingore last probability as it corresponds to 'end' token
        total_prob = 1
        for x in confidence[:-1]:
           total_prob = total_prob * float(x)

        all_results[tracklet].append([value, total_prob])

    final_full_results = {}
    for tracklet in all_results.keys():
        if len(all_results[tracklet]) == 0:
            continue
        results = all_results[tracklet]
        best_prediction, all_unique, weights = find_best_prediction_with_lstm(results, model, device, max_seq_len, vocab_size)

        final_results[tracklet] = str(int(best_prediction))
        final_full_results[tracklet] = {'label': str(int(best_prediction)), 'unique': all_unique, 'weights': weights}

    return final_results, final_full_results

def train():
    # Device configuration
    device = "cuda:0"
    print(f"Using device: {device}")

    # Create model
    model = PredictionConsolidatorLSTM(vocab_size, embedding_dim, hidden_dim, max_seq_len, bidirectional).to(device)

    # # Load Model if it exists:
    # if os.path.exists(model_save_path):
    #     model.load_state_dict(torch.load(model_save_path))
    #     print("Model loaded successfully!")
    # else:
    #     print("No saved model found!")

    # Example Usage (replace 'your_file.json' and set your model parameters)

    # Create dummy data for training with a batch size of 32
    num_training_samples = 1210
    #   train_sequences = torch.rand(batch_size, max_seq_len, 2).to(device)  # random values for numbers and confidences.
    #   train_sequences[:, :, 0] = torch.randint(0, vocab_size, (batch_size, max_seq_len)).float()  # random integer values for the token
    #   train_labels = torch.randint(0, 100, (batch_size,), dtype=torch.long).to(device)  # changed from randint(min_val, max_val) to integers between 0 to 100 representing the different classes
    all_results = pd.read_pickle("my_object")
    with open('test_gt.json', 'r') as file:
        train_labels = json.load(file)
    #replace -1 in the train_labels with 0 to match the model output
    for key in train_labels.keys():
        if train_labels[key] == -1:
            train_labels[key] = 0
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("train_labels", train_labels)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use cross entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_validation_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0
        total_train_correct = 0
        total_train_examples = 0
        TotalLabels = []
        TotalPredictions = []
        for tracklet in all_results.keys():
            if len(all_results[tracklet]) == 0:
                continue
            results = torch.tensor(all_results[tracklet]).to(device) 
            optimizer.zero_grad()  # zero gradients
            output = model(results)  # shape: (1, 100) - batch size of 1
            predictions = model.get_prediction(output)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("train_labels", train_labels[tracklet])
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            label = torch.tensor(train_labels[tracklet]).to(device)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("label", label)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            one_hot_labels = nn.functional.one_hot(label, num_classes=100).float().to(device)     
            loss = criterion(output, one_hot_labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_train_loss += loss.item()
            TotalLabels.append(label)
            TotalPredictions.append(predictions)

        total_train_correct += calculate_accuracy(TotalPredictions, TotalPredictions)
        total_train_examples += len(all_results.keys())

        avg_train_accuracy = total_train_correct / total_train_examples
        print(f"Epoch {epoch + 1}: Train Loss: {total_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    #save the model
    torch.save(model.state_dict(), model_save_path)

    print("Finished training")

def process_jersey_id_predictions(str_result_file):

    # Create model
    model = PredictionConsolidatorLSTM(vocab_size, embedding_dim, hidden_dim, max_seq_len, bidirectional).to(device)

    # Load Model if it exists:
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("Model loaded successfully!")
    else:
        print("No saved model found!")

    final_results, final_full_results = process_jersey_id_predictions_Internal(str_result_file, model,device,max_seq_len,vocab_size)
    print("Final Results:", final_results)
    print("Final Full Results:", final_full_results)
    return final_results, final_full_results

#train()