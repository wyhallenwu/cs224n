import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

# Our raw data, which consists of sentences
corpus = [
    "We always come to Paris", "The professor is from Australia",
    "I live in Stanford", "He comes from Taiwan",
    "The capital of Turkey is Ankara"
]

# Set of locations that appear in our corpus
locations = set(
    ["australia", "ankara", "paris", "stanford", "taiwan",
     "turkey"])  # locations in our set


def preprocess_sentence(sentence):
    return sentence.lower().split()


# Given a sentence of tokens, return the corresponding indices
def convert_token_to_indices(sentence, word_to_ix):
    indices = []
    for token in sentence:
        # Check if the token is in our vocabularly. If it is, get it's index.
        # If not, get the index for the unknown token.
        if token in word_to_ix:
            index = word_to_ix[token]
        else:
            index = word_to_ix["<unk>"]
        indices.append(index)
    return indices


def pad_window(sentence, window_size, pad_token="<pad>"):
    window = [pad_token] * window_size
    return window + sentence + window


def custom_collate_fn(batch, window_size, word_to_ix):
    # prepare the datapoints
    x, y = zip(*batch)
    x = [pad_window(s, window_size=window_size) for s in x]
    x = [convert_token_to_indices(s, word_to_ix=word_to_ix) for s in x]

    # pad x so that all the examples have the same size
    pad_token_ix = word_to_ix["<pad>"]
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = nn.utils.rnn.pad_sequence(x,
                                         batch_first=True,
                                         padding_value=pad_token_ix)

    # pad y and record the length
    lengths = [len(label) for label in y]
    lengths = torch.LongTensor(lengths)
    y = [torch.LongTensor(y_i) for y_i in y]
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return x_padded, y_padded, lengths


# counter = 0
# for batched_x, batched_y, batched_lengths in loader:
#     print(f"Iteration {counter}")
#     print("Batched Input:")
#     print(batched_x)
#     print("Batched Labels:")
#     print(batched_y)
#     print("Batched Lengths:")
#     print(batched_lengths)
#     print("")
#     counter += 1


class WordWindowClassifier(nn.Module):

    def __init__(self, hyperparameters, vocab_size, pad_ix=0):
        super(WordWindowClassifier, self).__init__()

        self.window_size = hyperparameters["window_size"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]

        self.embeds = nn.Embedding(vocab_size,
                                   self.embed_dim,
                                   padding_idx=pad_ix)
        if self.freeze_embeddings:
            self.embed_layer.weight.requires_grad = False

        full_window_size = 2 * window_size + 1
        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
            nn.Tanh())
        self.output_layer = nn.Linear(self.hidden_dim, 1)

        self.probabilities = nn.Sigmoid()

    def forward(self, inputs):
        """
        Let B:= batch_size
            L:= window-padded sentence length
            D:= self.embed_dim
            S:= self.window_size
            H:= self.hidden_dim
        
        inputs: a (B, L) tensor of token indices
        """
        # input (B, L) -> (B, L~, S)
        B, L = inputs.size()
        token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)
        _, adjusted_length, _ = token_windows.size()

        # Good idea to do internal tensor-size sanity checks, at the least in comments!
        assert token_windows.size() == (B, adjusted_length,
                                        2 * self.window_size + 1)

        # (B, L~, S) -> (B, L~, S, D)
        embedded_windows = self.embeds(token_windows)
        # (B, L~, S, D) -> (B, L~, S*D)
        embedded_windows = embedded_windows.view(B, adjusted_length, -1)
        # layer1 (B, L~, S*D) -> (B, L~, H)
        layer_1 = self.hidden_layer(embedded_windows)
        # layer2 (B, L~, H) -> (B, L~, 1)
        output = self.output_layer(layer_1)
        # (B, L~, 1)
        output = self.probabilities(output)
        output = output.view(B, -1)

        return output


def loss_function(batch_outputs, batch_labels, batch_lengths):
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())
    loss = loss / batch_lengths.sum().float()

    return loss


# Function that will be called in every epoch
def train_epoch(loss_function, optimizer, model, loader):

    # Keep track of the total loss for the batch
    total_loss = 0
    for batch_inputs, batch_labels, batch_lengths in loader:
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        outputs = model.forward(batch_inputs)
        # Compute the batch loss
        loss = loss_function(outputs, batch_labels, batch_lengths)
        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss


# Function containing our main training loop
def train(loss_function, optimizer, model, loader, num_epochs=10000):

    # Iterate through each epoch and call our train_epoch function
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(loss_function, optimizer, model, loader)
        if epoch % 100 == 0: print(epoch_loss)


if __name__ == '__main__':
    # Create our training set
    train_sentences = [sent.lower().split() for sent in corpus]

    # Set of locations that appear in our corpus
    locations = set(
        ["australia", "ankara", "paris", "stanford", "taiwan", "turkey"])

    # Our train labels
    train_labels = [[1 if word in locations else 0 for word in sent]
                    for sent in train_sentences]

    # Find all the unique words in our corpus
    vocabulary = set(w for s in train_sentences for w in s)
    # Add the unknown token to our vocabulary
    vocabulary.add("<unk>")
    # Add the <pad> token to our vocabulary
    vocabulary.add("<pad>")

    ix_to_word = sorted(list(vocabulary))

    # Creating a dictionary to find the index of a given word
    word_to_ix = {word: ind for ind, word in enumerate(ix_to_word)}

    data = list(zip(train_sentences, train_labels))
    batch_size = 2
    shuffle = True
    window_size = 2
    collate_fn = partial(custom_collate_fn,
                         window_size=window_size,
                         word_to_ix=word_to_ix)

    loader = DataLoader(data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn)

    # Initialize a model
    # It is useful to put all the model hyperparameters in a dictionary
    model_hyperparameters = {
        "batch_size": 4,
        "window_size": 2,
        "embed_dim": 25,
        "hidden_dim": 25,
        "freeze_embeddings": False,
    }

    vocab_size = len(word_to_ix)
    model = WordWindowClassifier(model_hyperparameters, vocab_size)

    # Define an optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 10000
    train(loss_function, optimizer, model, loader, num_epochs=num_epochs)

    # Create test sentences
    test_corpus = ["She comes from Paris"]
    test_sentences = [s.lower().split() for s in test_corpus]
    test_labels = [[0, 0, 0, 1]]

    # Create a test loader
    test_data = list(zip(test_sentences, test_labels))
    batch_size = 1
    shuffle = False
    window_size = 2
    collate_fn = partial(custom_collate_fn,
                         window_size=2,
                         word_to_ix=word_to_ix)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=collate_fn)

    for test_instance, labels, _ in test_loader:
        outputs = model.forward(test_instance)
        print(labels)
        print(outputs)