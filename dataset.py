import mido
from mido import MidiFile, MidiTrack, Message

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Function to read MIDI file
def read_midi_file(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
        # You can process the MIDI file here
        time_since_start = 0
        ret = []
        for track in mid.tracks:
            time_since_start = 0
            for msg in track:
                time_since_start += msg.time
                if msg.type == 'note_on':
                    note = msg.note
                    velocity = msg.velocity
                    start_time = time_since_start
                    ret.append((start_time, note))
        return ret
    except IOError:
        print(f"Error reading {midi_path}")

def adjust_sequence_length(sequence, target_length=1000):
    # Convert the sequence to a tensor
    seq_tensor = torch.tensor(sequence, dtype=torch.float32)

    # Truncate if the sequence is longer than the target length
    if seq_tensor.size(0) > target_length:
        return seq_tensor[:target_length]

    # Pad with zeros if the sequence is shorter than the target length
    elif seq_tensor.size(0) < target_length:
        padding_size = target_length - seq_tensor.size(0)
        padding = torch.zeros((padding_size, seq_tensor.size(1)), dtype=torch.float32)
        return torch.cat((seq_tensor, padding), dim=0)

    return seq_tensor

# Replace with the path to your text file
text_file_path = 'dataset/single_track_music.txt'
train_data = []

with open(text_file_path, 'r') as file:
    for line in file:
        midi_file_path = line.strip()  # Remove any trailing newline characters
        data_tensor = torch.tensor(read_midi_file(midi_file_path))
        train_data.append(adjust_sequence_length(data_tensor))

train_data = torch.stack(train_data)
print(train_data)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward propagate the LSTM
        out, _ = self.lstm(x)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

class MusicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming you want to predict the next note based on previous ones
        # Input: All notes except the last one
        # Target: The last note
        input_sequence = self.data[idx, :-1, :]
        target_sequence = self.data[idx, -1, :]
        return input_sequence, target_sequence


input_size = 2  # Number of input features (time_since_start, note)
hidden_size = 128  # Size of hidden layer
num_layers = 2  # Number of LSTM layers
output_size = 2  # Output size (predicting the next (time_since_start, note) pair)

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

music_dataset = MusicDataset(train_data)

batch_size = 4  # Adjust as needed
shuffle = True  # Shuffle the data every epoch
num_workers = 0  # Number of subprocesses for data loading

music_dataloader = DataLoader(music_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Parameters
num_epochs = 100  # Number of epochs
learning_rate = 0.1  # Learning rate

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for a regression task
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0

    for inputs, targets in music_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item() * inputs.size(0)

    # Calculate average loss for the epoch
    epoch_loss /= len(music_dataloader.dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def generate_sequence(model, start_sequence, num_notes):
    model.eval()  # Set the model to evaluation mode
    current_sequence = start_sequence.clone().detach()

    generated_sequence = []
    for _ in range(num_notes):
        with torch.no_grad():
            # Predict the next note
            next_note = model(current_sequence).unsqueeze(0)
            # Add the predicted note to the sequence
            generated_sequence.append(next_note.squeeze().cpu().numpy())
            # Use the updated sequence for the next prediction
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_note), dim=1)

    return generated_sequence

def generate_sequence(model, start_sequence, num_notes):
    model.eval()  # Set the model to evaluation mode
    current_sequence = start_sequence.clone().detach()

    generated_sequence = []
    for _ in range(num_notes):
        with torch.no_grad():
            # Predict the next note
            print(current_sequence)
            next_note = model(current_sequence).unsqueeze(0)
            # Add the predicted note to the sequence
            generated_sequence.append(next_note.squeeze().cpu().numpy())
            # Use the updated sequence for the next prediction
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_note), dim=1)

    return generated_sequence

def create_midi(generated_sequence, output_file='generated_song.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for time_since_start, note in generated_sequence:
        # Convert to integer values if they are not
        note = int(note)
        time_since_start = int(time_since_start)

        # Add note_on and note_off events
        track.append(Message('note_on', note=note, velocity=64, time=time_since_start))
        track.append(Message('note_off', note=note, velocity=127, time=time_since_start + 480))  # Adjust time as needed

    mid.save(output_file)


# Select a sequence
sequence_index = 0  # Index of the sequence to use
prefix_length = 50  # Length of the prefix

# Select the prefix
prefix = train_data[sequence_index, :prefix_length, :]
prefix = prefix.unsqueeze(0)  # Add batch dimension
print(prefix)

num_notes_to_generate = 10  # Number of new notes to generate
generated_sequence = torch.tensor(generate_sequence(model, prefix, num_notes_to_generate))
print(generated_sequence)

# create_midi(generated_sequence, output_file='generated_song.mid')
