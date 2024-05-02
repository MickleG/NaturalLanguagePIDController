import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import gym
import pybulletgym
import math
import pybullet as p
import gensim.downloader as api

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Constants/hyperparameters

model_type = 'magtype'

full_dataset_path = "PID_dataset.txt"
model_path = "best_model_" + model_type + ".pth"

label_mapping_gaintype = {
    'p': 0,
    'i': 1,
    'd': 2
}

label_mapping_magtype = {
    'large increase': 0,
    'small increase': 1,
    'no change': 2,
    'small decrease': 3,
    'large decrease': 4
}

# Data splitting parameters
train_percentage = 0.7
val_percentage = 0.2

# Embedding parameter (set to 50 for glove embeddings)
embedding_dim = 50

# Network Hyperparameters
batch_size_gaintype = 64
batch_size_magtype = 128

hidden_dim_gaintype = 1024
hidden_dim_magtype = 256

learning_rate = 0.00001
num_epochs_gaintype = 10
num_epochs_magtype = 10

num_layers_gaintype = 1
num_layers_magtype = 2

dropout_rate_gaintype = 0
dropout_rate_magtype = 0.4


# Defining output size based on classifier type
output_dim_gaintype = 3
output_dim_magtype = 5

# Parameter for bidirectionality
do_bidirectional_gaintype = False
do_bidirectional_magtype = True
do_lr_scheduling = True

train = False

full_testing = True



if torch.backends.mps.is_available():
    print("MPS is available! Training on GPU...")
    torch_device = torch.device("mps")
else:
    print("MPS is not available. Training on CPU...")
    torch_device = torch.device("cpu")

# Defining classes
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab or self.build_vocab(self.texts)

    def build_vocab(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split())
        vocab = {word: i for i, word in enumerate(sorted(vocab))}
        return vocab

    def text_to_tensor(self, text):
        return torch.tensor([self.vocab[word] for word in text.split() if word in self.vocab], dtype=torch.long)


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.text_to_tensor(self.texts[idx])
        label = int(self.labels[idx])
        return text, label
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate, embeddings, bidirectional, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embeddings is None else nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim * 1, output_dim)
    
    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, (hidden, _) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        output = self.fc(hidden)
        return torch.softmax(output, dim=1)

# Defining helper functions
def load_and_process_data(file_path, train_percentage, val_percentage, classifier_type):
    inputs = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            content, label = line.split('(')
            inputs.append(content.strip())

            if(classifier_type == 'gaintype'):
              labels.append(label_mapping_gaintype[label[0]])
            else:
              label = label.split(', ')[1]
              labels.append(label_mapping_magtype[label.strip(')\n').strip()])
    
    # Shuffle data
    data = list(zip(inputs, labels))
    random.shuffle(data)
    inputs, labels = zip(*data)

    # Split data
    total = len(inputs)
    train_end = int(total * train_percentage)
    val_end = train_end + int(total * val_percentage)
    
    train_inputs, train_labels = inputs[:train_end], labels[:train_end]
    val_inputs, val_labels = inputs[train_end:val_end], labels[train_end:val_end]
    test_inputs, test_labels = inputs[val_end:], labels[val_end:]
    
    return (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels)
def collate_batch(batch):
    sentence_batch, label_batch = zip(*batch)
    sentence_batch = pad_sequence([s.clone().detach() for s in sentence_batch], padding_value=0, batch_first=True)
    label_batch = torch.tensor(label_batch, dtype=torch.long).flatten()
    return sentence_batch, label_batch
def create_embeddings(vocabulary):
  glove_embs = api.load("glove-wiki-gigaword-50")
  embeddings = torch.zeros(len(vocabulary), embedding_dim)

  for word, idx in vocabulary_gaintype.items():
      if word in glove_embs:
          embeddings[idx] = torch.tensor(glove_embs[word])
      else:
          embeddings[idx] = torch.randn(embedding_dim)

  return embeddings
def train_model(model, data_loader, val_loader, loss_fn, optimizer, num_epochs):
    global learning_rate

    losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    no_best_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        model.train()

        total = len(data_loader.dataset)
        epoch_losses = []

        for i, (sentence, label) in enumerate(data_loader):
            optimizer.zero_grad()

            sentence = sentence.to(torch_device)
            label = label.to(torch_device)

            predictions = model(sentence)


            loss = loss_fn(predictions, label)

            loss = loss.cpu()

            epoch_losses.append(loss.detach().numpy())


            loss.backward()
            optimizer.step()

            done = int(50 * (i + 1) / len(data_loader))
            print(f"\r[{'=' * done}{' ' * (50-done)}] {i + 1}/{len(data_loader)} batches", end='')

        print()
        train_accuracy = evaluate_model(model, data_loader)
        val_accuracy = evaluate_model(model, val_loader)
        train_accuracies.append(train_accuracy)

        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path} with Validation Accuracy: {val_accuracy:.4f}')
        else:
            no_best_counter += 1

        if(no_best_counter >= 2 and do_lr_scheduling):
                learning_rate /= 10
                print("Updating learning rate to {}".format(learning_rate))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                no_best_counter = 0

        losses.append(np.mean(epoch_losses))

    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.show()

    plt.plot(range(1, num_epochs + 1), train_accuracies, label="train accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.legend()
    plt.show()
def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for sentence, label in data_loader:
            sentence = sentence.to(torch_device)
            label = label.to(torch_device)
            predictions = model(sentence)
            _, predicted_label = predictions.max(1)  # Get the index of the max log-probability
            predicted_label = predicted_label.cpu()
            label = label.cpu()
            total_correct += (predicted_label == label).sum().item()
            total_samples += label.size(0)

    accuracy = total_correct / total_samples
    return accuracy
def predict_class(model, text, dataset, model_type):
    if(model_type == 'gaintype'):
        decoding_dict = label_mapping_gaintype
    else:
        decoding_dict = label_mapping_magtype

    if text.strip() == "":
        return None  # Exit condition for empty input
    text_tensor = dataset.text_to_tensor(text).unsqueeze(0)  # Add batch dimension
    text_tensor = text_tensor.to(torch_device)  # Move to the same device as model
    with torch.no_grad():
        model.eval()
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    return output, list(decoding_dict.keys())[predicted_class]

def pid_control(target, current, prev_error, integral, gains):
    kp = gains[0]
    ki = gains[1]
    kd = gains[2]
    error = np.subtract(target, current)
    integral += error
    derivative = np.subtract(error, prev_error)
    output = kp * error + ki * integral + kd * derivative

    # print("output is: {} {} {}".format(kp*error, ki*integral, kd*derivative))
    return output, error, integral
def ik(xy_cart):
    x = xy_cart[0]
    y = xy_cart[1]

    r1 = np.sqrt(x**2 + y**2)
    phi_1 = np.arccos((l0**2 - l1**2 - r1**2) / (-2 * l1 * r1))
    phi_2 = np.arctan2(y, x)
    theta_1 = phi_2 - phi_1

    phi_3 = np.arccos((r1**2 - l1**2 - l0**2) / (-2 * l0 * l1))
    theta_2 = math.pi - phi_3

    if theta_1 > np.pi:
        theta_1 -= 2*np.pi
    if theta_2 > np.pi:
        theta_2 -= 2*np.pi

    return [theta_1, theta_2]
def fuzzy_pid_gain_change(magtype, gaintype_activations):

    gaintype = np.argmax(gaintype_activations)

    new_gains = [0, 0, 0]

    if(magtype == 'large increase'):
        if(gaintype == 0):
            new_gains[0] += gaintype_activations[0] * 0.0015
        elif(gaintype == 1):
            new_gains[1] += gaintype_activations[1] * 0.00001
        else:
            new_gains[2] += gaintype_activations[2] * 0.1
    elif(magtype == 'small increase'):
        if(gaintype == 0):
            new_gains[0] += gaintype_activations[0] * 0.00005
        elif(gaintype == 1):
            new_gains[1] += gaintype_activations[1] * 0.000001
        else:
            new_gains[2] += gaintype_activations[2] * 0.05
    elif(magtype == 'large decrease'):
        if(gaintype == 0):
            new_gains[0] -= gaintype_activations[0] * 0.0015
        elif(gaintype == 1):
            new_gains[1] -= gaintype_activations[1] * 0.00001
        else:
            new_gains[2] -= gaintype_activations[2] * 0.1
    elif(magtype == 'small decrease'):
        if(gaintype == 0):
            new_gains[0] -= gaintype_activations[0] * 0.00005
        elif(gaintype == 1):
            new_gains[1] -= gaintype_activations[1] * 0.000001
        else:
            new_gains[2] -= gaintype_activations[2] * 0.05


    return new_gains
def controller():
    integral0 = 0
    integral1 = 0
    prev_error0 = 0
    prev_error1 = 0
    error0 = 0
    error1 = 0


    joint0_history = []
    joint1_history = []

    joint0_target = []
    joint1_target = []

    ee_pos_x = []
    ee_pos_y = []

    error0_history = []
    error1_history = []

    counter = 0

    while(counter < 50000):
        target_position = [math.pi / 6, math.pi / 4]

        q0, q0_dot = env.unwrapped.robot.central_joint.current_position()
        q1, q1_dot = env.unwrapped.robot.elbow_joint.current_position()

        current_position = [q0, q1]
        
        control_signal0, prev_error0, integral0 = pid_control(target_position[0], current_position[0], prev_error0, integral0, gains0)
        control_signal1, prev_error1, integral1 = pid_control(target_position[1], current_position[1], prev_error1, integral1, gains1)

       
        scaled_control_signal0 = np.clip(control_signal0, -1, 1)
        scaled_control_signal1 = np.clip(control_signal1, -1, 1)

        obs, reward, done, _ = env.step([scaled_control_signal0, scaled_control_signal1])

        env.render()

        env.unwrapped._p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=0,
            cameraPitch=-80,
            cameraTargetPosition=[0, 0, camera_distance]
        )


        ee_x = l0 * math.cos(current_position[0]) + l1 * math.cos(current_position[0] + current_position[1])
        ee_y = l0 * math.sin(current_position[0]) + l1 * math.sin(current_position[0] + current_position[1])

        joint0_history.append(current_position[0])
        joint1_history.append(current_position[1])

        joint0_target.append(target_position[0])
        joint1_target.append(target_position[1])

        error0_history.append(prev_error0)
        error1_history.append(prev_error1)

        ee_pos_x.append(ee_x)
        ee_pos_y.append(ee_y)

        counter += 1

    time = range(len(error0_history))
    plt.plot(time, joint0_target, label="target0")
    plt.plot(time, joint0_history, label="actual0")

    plt.plot(time, joint1_target, label="target1")
    plt.plot(time, joint1_history, label="actual1")

    plt.title("Individual Joint Configuration Comparison")
    plt.xlabel("Time (Simulation Timesteps)")
    plt.ylabel("Joint Angle (rad)")

    plt.legend()
    plt.show()

    plt.title("Individual Configuration Error")
    plt.xlabel("Time (Simulation Timesteps)")
    plt.ylabel("Error")
    plt.plot(time, error0_history, label="error0")
    plt.plot(time, error1_history, label="error1")

    plt.legend()
    plt.show()


# Data splitting (variables ending in gaintype are for the gain classifier, variables ending in magtype are for magnitude classifier)
train_data_gaintype, val_data_gaintype, test_data_gaintype = load_and_process_data(file_path=full_dataset_path, train_percentage=train_percentage, val_percentage=val_percentage, classifier_type='gaintype')
train_data_magtype, val_data_magtype, test_data_magtype = load_and_process_data(file_path=full_dataset_path, train_percentage=train_percentage, val_percentage=val_percentage, classifier_type='magtype')

train_dataset_gaintype = TextDataset(*train_data_gaintype)
val_dataset_gaintype = TextDataset(*val_data_gaintype)
test_dataset_gaintype = TextDataset(*test_data_gaintype)

train_dataset_magtype = TextDataset(*train_data_magtype)
val_dataset_magtype = TextDataset(*val_data_magtype)
test_dataset_magtype = TextDataset(*test_data_magtype)

train_loader_gaintype = DataLoader(train_dataset_gaintype, batch_size=batch_size_gaintype, shuffle=True, collate_fn=collate_batch)
val_loader_gaintype = DataLoader(val_dataset_gaintype, batch_size=batch_size_gaintype, shuffle=False, collate_fn=collate_batch)
test_loader_gaintype = DataLoader(test_dataset_gaintype, batch_size=batch_size_gaintype, shuffle=False, collate_fn=collate_batch)

train_loader_magtype = DataLoader(train_dataset_magtype, batch_size=batch_size_magtype, shuffle=True, collate_fn=collate_batch)
val_loader_magtype = DataLoader(val_dataset_magtype, batch_size=batch_size_magtype, shuffle=False, collate_fn=collate_batch)
test_loader_magtype = DataLoader(test_dataset_magtype, batch_size=batch_size_magtype, shuffle=False, collate_fn=collate_batch)

# Embeddings to feed into the network
vocabulary_gaintype = train_dataset_gaintype.vocab
vocabulary_magtype = train_dataset_magtype.vocab
embeddings_gaintype = create_embeddings(vocabulary_gaintype)
embeddings_magtype = create_embeddings(vocabulary_magtype)

print("Data loaded and split. Proceeding to train.")
print("Model is {}".format(model_type))

if(not full_testing):
    if(model_type == 'gaintype'):
        # Model training

        model_gaintype = LSTM(vocab_size=len(vocabulary_gaintype), embedding_dim=embedding_dim, hidden_dim=hidden_dim_gaintype, output_dim=output_dim_gaintype, dropout_rate=dropout_rate_gaintype, embeddings=embeddings_gaintype, bidirectional=do_bidirectional_gaintype, num_layers=num_layers_gaintype).to(torch_device)
        loss_fn_gaintype = nn.CrossEntropyLoss()
        optimizer_gaintype = torch.optim.Adam(model_gaintype.parameters(), lr=learning_rate)

        if(train == True):
            train_model(model_gaintype, train_loader_gaintype, val_loader_gaintype, loss_fn_gaintype, optimizer_gaintype, num_epochs_gaintype)
            print("Test accuracy is: {}".format(evaluate_model(model_gaintype, test_loader_gaintype)))

        model_gaintype.load_state_dict(torch.load(model_path))
        model_gaintype.eval()

        # User interaction loop
        print("Enter a string to classify, or an empty string to quit:")
        while True:
            user_input = input("Input: ")
            if user_input == "":
                print("Exiting...")
                break
            _, class_index = predict_class(model_gaintype, user_input, train_dataset_gaintype, 'gaintype')
            print(f"Predicted class: {class_index}")

    else:
        # Model training
        model_magtype = LSTM(vocab_size=len(vocabulary_magtype), embedding_dim=embedding_dim, hidden_dim=hidden_dim_magtype, output_dim=output_dim_magtype, dropout_rate=dropout_rate_magtype, embeddings=embeddings_magtype, bidirectional=do_bidirectional_magtype, num_layers=num_layers_magtype).to(torch_device)
        loss_fn_magtype = nn.CrossEntropyLoss()
        optimizer_magtype = torch.optim.Adam(model_magtype.parameters(), lr=learning_rate)

        if(train == True):
            train_model(model_magtype, train_loader_magtype, val_loader_magtype, loss_fn_magtype, optimizer_magtype, num_epochs_magtype)
            print("Test accuracy is: {}".format(evaluate_model(model_magtype, test_loader_magtype)))

        model_magtype.load_state_dict(torch.load(model_path))
        model_magtype.eval()

        # User interaction loop
        print("Enter a string to classify, or an empty string to quit:")
        while True:
            user_input = input("Input: ")
            if user_input == "":
                print("Exiting...")
                break
            _, class_index = predict_class(model_magtype, user_input, train_dataset_magtype, 'magtype')
            print(f"Predicted class: {class_index}")
else:
    env = gym.make('ReacherPyBulletEnv-v0')
    env.render(mode="human")
    obs = env.reset()

    physics_client = p.connect(p.DIRECT)

    num_bodies = p.getNumBodies()

    p.setGravity(0, 0, 0)


    # gains0_start = [0.15, 0, 7]
    # gains1_start = [0.2, 0, 12]
    gains0 = [0.000001, 0, 0.001]
    gains1 = [0.000001, 0, 0.001]

    l0 = 0.1
    l1 = 0.11

    camera_distance = 0.2

    model_gaintype = LSTM(vocab_size=len(vocabulary_gaintype), embedding_dim=embedding_dim, hidden_dim=hidden_dim_gaintype, output_dim=output_dim_gaintype, dropout_rate=dropout_rate_gaintype, embeddings=embeddings_gaintype, bidirectional=do_bidirectional_gaintype, num_layers=num_layers_gaintype).to(torch_device)
    model_magtype = LSTM(vocab_size=len(vocabulary_magtype), embedding_dim=embedding_dim, hidden_dim=hidden_dim_magtype, output_dim=output_dim_magtype, dropout_rate=dropout_rate_magtype, embeddings=embeddings_magtype, bidirectional=do_bidirectional_magtype, num_layers=num_layers_magtype).to(torch_device)

    model_gaintype.load_state_dict(torch.load("best_model_gaintype.pth"))
    model_gaintype.eval()

    model_magtype.load_state_dict(torch.load("best_model_magtype.pth"))
    model_magtype.eval()

    env.unwrapped.robot.central_joint.reset_position(0, 0)
    env.unwrapped.robot.elbow_joint.reset_position(0, 0)
    controller()
    env.reset()

    request_continue = input("Would you like to adjust the controller's performance? (y/n) ")

    while(request_continue == 'y'):
        request0 = input('Please input request for shoulder joint (yellow): ')
        request1 = input('Please input request for elbow joint (blue): ')

        gaintype_activations0, _ = predict_class(model_gaintype, request0, train_dataset_gaintype, 'gaintype')
        gaintype_activations1, _ = predict_class(model_gaintype, request0, train_dataset_gaintype, 'gaintype')
        _, result0_mag = predict_class(model_magtype, request1, train_dataset_magtype, 'magtype')
        _, result1_mag = predict_class(model_magtype, request1, train_dataset_magtype, 'magtype')

        gaintype_activations0 = gaintype_activations0.cpu().tolist()[0]
        gaintype_activations1 = gaintype_activations1.cpu().tolist()[0]

        print("Request for joint0: gaintype is {} and magtype is {}".format(gaintype_activations0, result0_mag))
        print("Request for joint1: gaintype is {} and magtype is {}".format(gaintype_activations1, result1_mag))

        gains_change0 = fuzzy_pid_gain_change(result0_mag, gaintype_activations0)
        gains_change1 = fuzzy_pid_gain_change(result1_mag, gaintype_activations1)

        gains0 = [max(0, g + d) for g, d in zip(gains0, gains_change0)]
        gains1 = [max(0, g + d) for g, d in zip(gains1, gains_change1)]

        print("new gains 0: {}".format(gains0))
        print("new gains 1: {}".format(gains1))

        env.unwrapped.robot.central_joint.reset_position(0, 0)
        env.unwrapped.robot.elbow_joint.reset_position(0, 0)
        controller()
        env.reset()

        request_continue = input("Would you like to adjust the controller's performance? (y/n) ")








