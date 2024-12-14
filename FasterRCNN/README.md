# Faster R-CNN

Digital image processing 2024

***Final progect: Smoking detection & Auto-pixelation***

Implementation by pre-trained Faster R-CNN on `torchvision`.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/link/of/this/repository
    ```
2. Navigate to the project directory:
    ```bash
    cd FasterRCNN
    ```

## Environment

This project was implemeated under **Anaconda3 virtual environment** on **Windows 11**.

1. Download Anaconda3 on your own device and type:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the virtual environment for pytorch:
    ```bash
    conda activate torch
    ```

## Usage

1. Run the script to download dataset:
    ```bash
    python load_dataset.py
    ```
2. Run the training and testing process by running:
    ```bash
    python main.py
    ```
3. Fine-tuned model is saved under `fine_tuned/`.

4. Visualize the pixelation on example image by running:
    ```bash
    python visualize.py
    ```

5. Visualize the pixelation on example video by running:
    ```bash
    python video.py
    ```

## Example

|Origin|Pixelated|
|----------|----------|
| ![Original Image](example/example.jpg) | ![Pixelated Image](example/example_pixelated.jpg) |
<!-- | ![Origin Video](example/example.gif) | ![Origin Video](example/example.gif) | -->

