# Multilingual OCR Detection

## Architecture Diagram
![Architecture Diagram](path_to_your_architecture_diagram.png)

## Features
- Multilingual text detection
- EAST (Efficient and Accurate Scene Text detection) model
- Text grouping based on proximity
- Language classification for detected text
- TrOCR (Transformer-based Optical Character Recognition) for text recognition

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ripuns/Multilingual-OCR-Detection.git
   cd Multilingual-OCR-Detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your input images.
2. Run the detection script:
   ```bash
   python detect.py --image_path path_to_your_image.jpg
   ```
3. The output will include the detected text along with its language.

## Project Structure
- `detect.py`: Main script for running the detection.
- `models/`: Contains pre-trained models like EAST and TrOCR.
- `utils/`: Utility functions for processing and visualizing results.
- `data/`: Sample images and datasets.

## Technical Details
The multilingual OCR detection system utilizes:
- **EAST Detector**: For locating text regions in images.
- **Text Grouping**: To cluster text boxes that are close to each other, improving detection accuracy.
- **Language Classification**: To identify the language of the detected text.
- **TrOCR Recognition**: A powerful OCR engine that can recognize text from images in various languages, transforming visual data into machine-readable text.

### Conclusion
This project provides an efficient solution for detecting and recognizing text across multiple languages using a combination of advanced computer vision techniques and deep learning models.