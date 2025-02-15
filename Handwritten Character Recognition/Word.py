# Dataset Acquisition

!wget -q https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip
!unzip -qq IAM_Words.zip

!mkdir handwriting_data
!mkdir handwriting_data/samples
!tar -xf IAM_Words/words.tgz -C handwriting_data/samples
!mv IAM_Words/words.txt handwriting_data

!head -20 handwriting_data/words.txt

# Dependencies

from tensorflow.keras.layers import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

np.random.seed(2023)
tf.random.set_seed(2023)

dataset_root = "handwriting_data"
entries = []

raw_data = open(f"{dataset_root}/words.txt", "r").readlines()
for entry in raw_data:
    if entry[0] == "#":
        continue  # Skip comment lines
    if entry.split(" ")[1] != "err":  # Skip invalid entries
        entries.append(entry)

print(f"Total valid entries: {len(entries)}")

np.random.shuffle(entries)

# train-test-val splits

train_split = int(0.85 * len(entries))
train_data = entries[:train_split]
remaining = entries[train_split:]

val_split = int(0.5 * len(remaining))
validation_data = remaining[:val_split]
test_data = remaining[val_split:]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Testing samples: {len(test_data)}")

samples_dir = os.path.join(dataset_root, "samples")

# Function to load sample data
def load_sample_data(data_entries):
    image_paths = []
    processed_entries = []
    for entry in data_entries:
        components = entry.strip().split(" ")
        filename = components[0]

        # Extract directory structure components
        dir_part1 = filename.split("-")[0]
        dir_part2 = f"{dir_part1}-{filename.split('-')[1]}"
        full_path = os.path.join(
            samples_dir,
            dir_part1,
            dir_part2,
            f"{filename}.png"
        )

        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            image_paths.append(full_path)
            processed_entries.append(components[-1].strip())

    return image_paths, processed_entries

# Loading
train_images, train_texts = load_sample_data(train_data)
val_images, val_texts = load_sample_data(validation_data)
test_images, test_texts = load_sample_data(test_data)

# Character Vocabulary Setup
unique_chars = set()
max_text_length = 0

# Analyze training texts
for text in train_texts:
    unique_chars.update(list(text))
    max_text_length = max(max_text_length, len(text))

unique_chars = sorted(list(unique_chars))
vocab_size = len(unique_chars)

print(f"Longest text sequence: {max_text_length}")
print(f"Character vocabulary: {vocab_size}")

def process_text_labels(labels):
    return [label.split(" ")[-1].strip() for label in labels]

val_texts = process_text_labels(val_texts)
test_texts = process_text_labels(test_texts)

# Create encoding/decoding layers
char_encoder = StringLookup(
    vocabulary=unique_chars,
    mask_token=None
)

char_decoder = StringLookup(
    vocabulary=char_encoder.get_vocabulary(),
    invert=True,
    mask_token=None
)

# Image Preprocessing
def preserve_aspect_resize(image, target_dims):
    target_w, target_h = target_dims
    image = tf.image.resize(image, (target_h, target_w), preserve_aspect_ratio=True)

    pad_vert = target_h - tf.shape(image)[0]
    pad_horz = target_w - tf.shape(image)[1]

    # Apply symmetric padding
    image = tf.pad(
        image,
        [
            [pad_vert // 2, pad_vert - pad_vert // 2],
            [pad_horz // 2, pad_horz - pad_horz // 2],
            [0, 0]
        ]
    )

    return image

BATCH_SIZE = 72
IMG_WIDTH = 144
IMG_HEIGHT = 36
PAD_TOKEN = 0 

# Function to prepare images
def prepare_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = preserve_aspect_resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return tf.cast(image, tf.float32) / 255.0

# Function to encode text labels
def encode_text(label):
    encoded = char_encoder(tf.strings.unicode_split(label, "UTF-8"))
    padding = max_text_length - tf.shape(encoded)[0]
    return tf.pad(encoded, [[0, padding]], constant_values=PAD_TOKEN)

# Function to create dataset
def create_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        lambda img, lbl: {
            "image": prepare_image(img),
            "label": encode_text(lbl)
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

training_ds = create_dataset(train_images, train_texts)
validation_ds = create_dataset(val_images, val_texts)
testing_ds = create_dataset(test_images, test_texts)


# Function to display sample images and labels
def display_samples(dataset, num_samples=8):
    data_batch = next(iter(dataset.take(1)))
    images = data_batch["image"]
    labels = data_batch["label"]

    plt.figure(figsize=(12, 9))
    for i in range(num_samples):
        ax = plt.subplot(3, 3, i + 1)

        # Process image
        img = images[i].numpy()
        img = np.squeeze(img, axis=-1)
        img = (img * 255).astype(np.uint8)

        # Decode label
        label = labels[i]
        label_chars = [c for c in label if c != PAD_TOKEN]
        decoded_text = tf.strings.reduce_join(
            char_decoder(label_chars)
        ).numpy().decode()

        ax.imshow(img, cmap="viridis")
        ax.set_title(decoded_text)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

display_samples(training_ds)
