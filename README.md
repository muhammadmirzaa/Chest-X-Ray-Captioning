# Chest-X-Ray-Captioning
This project uses Deep Learning models to generate radiology descriptions for Chest X Ray.

There are two aspects of the project. Initially an X-Ray image is passed through a pretrained neural network called ResNet 50.

The ResNet 50 was further trained on the CheXpert dataset but a smaller version.

The Neural Network detects some symptoms in the X-Ray and passes them on to the LLM which is finetuned on a medical transcription dataset.

The LLM used here is llama 3.2 1B by meta.
