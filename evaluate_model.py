import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure plot directory exists
os.makedirs('static/plots', exist_ok=True)

# Set paths
MODEL_PATH = 'trained_model.h5'
TEST_DIR = 'dataset/test'
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32

# Load the trained model
model = load_model(MODEL_PATH)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

# Get predictions
pred_probs = model.predict(test_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('static/plots/confusion_matrix.png')
plt.close()

# Classification report
report = classification_report(
    true_labels, pred_labels, output_dict=True, target_names=class_labels)
print(classification_report(true_labels, pred_labels, target_names=class_labels))

# Bar chart of precision, recall, f1-score
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    values = [report[label][metric] for label in class_labels]
    plt.bar(class_labels, values)
    plt.title(f'{metric.capitalize()} by Class')
    plt.ylim(0, 1)
    plt.ylabel(metric.capitalize())
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.savefig(f'static/plots/{metric}_by_class.png')
    plt.close()

print("✅ Evaluation complete. Charts saved in static/plots/")
