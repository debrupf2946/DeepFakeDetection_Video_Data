

# Deepfake Detection Research on Video Data

## Project Overview

This project focuses on deepfake detection on video data by implementing and evaluating various research papers. The goal is to develop a robust model capable of identifying manipulated videos and assessing the accuracy of these models using standard benchmarks and performance metrics.

## Table of Contents

- [Deepfake Detection Research on Video Data](#deepfake-detection-research-on-video-data)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Research Papers Implemented](#research-papers-implemented)
  - [Data](#data)
    - [Datasets](#datasets)
    - [Data Preprocessing](#data-preprocessing)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Research Papers Implemented

We are implementing and testing the following research papers on deepfake detection:

- [Deepfake Detection Based on the Adaptive Fusion of Spatial-Frequency Features :](https://onlinelibrary.wiley.com/doi/10.1155/2024/7578036)  
  - Detecting deepfake media remains an ongoing challenge, particularly as forgery techniques rapidly evolve and become increasingly diverse. Existing face forgery detection models typically attempt to discriminate fake images by identifying either spatial artifacts (e.g., generative distortions and blending inconsistencies) or predominantly frequency-based artifacts (e.g., GAN fingerprints). However, a singular focus on a single type of forgery cue can lead to limited model performance. In this work, we propose a novel cross-domain approach that leverages a combination of both spatial and frequency-aware cues to enhance deepfake detection. First, we extract wavelet features using wavelet transformation and residual features using a specialized frequency domain filter. These complementary feature representations are then concatenated to obtain a composite frequency domain feature set. Furthermore, we introduce an adaptive feature fusion module that integrates the RGB color features of the image with the composite frequency domain features, resulting in a rich, multifaceted set of classification features. Extensive experiments conducted on benchmark deepfake detection datasets demonstrate the effectiveness of our method. Notably, the accuracy of our method on the challenging FF++ dataset is mostly above 98%, showcasing its strong performance in reliably identifying deepfake images across diverse forgery techniques.


## Data

### Datasets

This project utilizes publicly available deepfake video datasets for training and testing the models:

- **[DFD](URL)**: The dataset contains over 100,000 video clips, including both real and deepfake samples. The deepfake videos were generated using state-of-the-art techniques such as FaceSwap and Face2Face, covering a diverse range of subjects, poses, and scenarios. 

- **[FaceForensics++](URL)**: is a large scale and widely used public facial forgery database that consists of 1000 real portrait videos and 1000 manipulated videos for each manipulation type. Most real portrait videos are collected from YouTube with the consent of the subjects. Each real video is manipulated by four manipulation methods, including DeepFake, FaceSwap, Face2Face, and NeuralTexture

- **[Celeb-DF](URL)**:This database contains 590 real videos extracted from YouTube, with a variety of diversity. These videos exhibit an extensive range of aspects, such as face sizes, lighting conditions, and backgrounds. As for fake videos, a total of 5639 videos are created swapping faces using DeepFake technology.

  
### Data Preprocessing

- Explanation of the preprocessing steps (e.g., frame extraction, normalization, resizing).
- Any transformations performed on the raw video data (e.g., converting to grayscale, frame extraction frequency).

<!--## Model Implementation-->

<!--The project includes the following models based on the research papers:-->

<!--- **Model 1**: Description of the architecture and approach.-->
<!--- **Model 2**: Description of the architecture and approach.-->
  
<!--We have used various techniques such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers to build and test the models.-->

<!--### Libraries and Frameworks-->

<!--- PyTorch,Keras 3.0-->
<!--- OpenCV-->
<!--- NumPy-->
<!--- Pandas-->
<!--- Scikit-learn-->
<!--- Matplotlib-->

## Installation

To set up the environment and dependencies for this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/deepfake-detection.git
    cd deepfake-detection
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the deepfake detection model, use the following commands:

1. Preprocess the data:
    ```bash
    python preprocess.py --input /path/to/video/files --output /path/to/output
    ```
   
2. Train a model:
    ```bash
    python train.py --model model_name --data /path/to/preprocessed/data
    ```

3. Evaluate a model:
    ```bash
    python evaluate.py --model model_name --test_data /path/to/test/data
    ```

4. For predictions:
    ```bash
    python predict.py --model model_name --video /path/to/input/video
    ```

## Evaluation

We evaluate the models using standard performance metrics such as:

- **Accuracy**
- **ROC AUC**

<!--We report the evaluation results in the following formats:-->

<!--- **Confusion Matrix**-->
<!--- **Precision-Recall Curve**-->
<!--- **ROC Curve**-->

## Results

| Model | Accuracy  | AUC |
|-------|----------|------|
| Model 1 | 85% |  0.91 |
| Model 2 | 78% |  0.85 |



## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

Please make sure your code follows the style guide and passes the tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.