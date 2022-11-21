# Soundscape Emotion Recognition (SER) Application
Master thesis for my NYU Music Technology degree. This application extracts audio features of audio files, predicts emotions of audio files, and plots predictions onto a 2-D emotion plane.

The motivation comes from the struggles of being a sound editor for films having to find sound effects with specific emotions and the lack of tools in the industry to search for the right sound effects.

---
## 1. Overview
- **SER_app/**: folder containing electron application
- **SER_model/**: folder containing machine learning model comparison & training
- **Thesis Paper**: Thesis paper itself for details of methodology 


## 2. Electron Application
### A. Feature Extraction
Extracted features for predictions:

- RMS
- Zero Crossing Rate
- Spectral Rolloff
- Spectral Centroid
- Spectral Spread
- Spectral Skewness
- Spectral Kurtosis
- Spectral Flatness
- Mel-Frequency Cepstral Coefficients (MFCCs) 13
- Chromagram
- Loudness
- Energy
- Perceptual Sharpness
- Spectral Slope

Features were extracted using Meyda.js immediately after selecting audio files and cached inside a JavaScript array to later be normalized and passed into the machine learning pipeline.

The bag-of-frame approach were used here to represent long-term statistical distribution of sequential audio data to save computation power and allow faster prediction with the model.

### B. Model Deployment
TensorFlow.js were used to load pretrained models and transform audio features into the correct input tensor shape (1,74) inside the application.

Models were saved as .h5 files and later converted into JSON and .bin files after the training process.

### C. Data Visualization & Interaction
Visualize predicted emotions of audio files inside 2-D valance-arousal emotion space using D3.js with mouse hover functionality of showing filenames and playing audio files.

![application interface](/img/App%20Interface.png)

- Valance (x-axis): unpleasent - pleasent (left - right)
- Arousal (y-axis): uneventful - eventful (left - right)


## 3. Model Comparison & Training
TensorFlow and sci-kit learn were used to A/B test and train machine learning models. Pandas were used to load, transform, and analyze dataset and matplotlib were used to visualize data.

### A. Model Comparision & Evaluation
#### Support Vector Regression (SVR) Models
|       Arousal      |    Mean   |    Std    |
| :----------------: | :-------: | :-------: |
|        $R^2$       | 0.785556  | 0.026073  |
| Mean Squared Error | 0.071068  | 0.009651  |

|       Valence      |    Mean   |    Std    |
| :----------------: | :-------: | :-------: |
|        $R^2$       | 0.547271	 | 0.051466  |
| Mean Squared Error | 0.148914  | 0.011666  |

#### Neural Networks
|       Arousal      |    Mean   |    Std    |
| :----------------: | :-------: | :-------: |
|        $R^2$       | 0.730940  | 0.161783  |
| Mean Squared Error | 0.070679	 | 0.028536  |

|       Valence      |    Mean   |    Std    |
| :----------------: | :-------: | :-------: |
|        $R^2$       | 0.504163	 | 0.121443  |
| Mean Squared Error | 0.138414	 | 0.038270  |

SVRs and Neural Nets compared with K-fold cross validation. As seen here, neural networks were less accurate due to the small dataset and having correlated features.
Logistical reasons led to the conclusion that sacrificing model performance for the less acurate neural networks was the right decision for this thesis.

Input tensor shape: (1, 74)

### B. Training
![neural network architecture](/img/Model%20Summary.png)

A simple multilayered perceptron (MLP) architecture was chosen for the neural networks. No regularization terms were implemented here due to the small dataset, only dropout layers were implemented with 0.2 dropout rate.

Batch normalization should have been implemented in the pipeline here to normalize training and prediction data to allow faster learning and better performance but oh well, you live and learn.


## 4. Reference
- Emotion recognition data from Emo-Soundscapes: https://metacreation.net/emo-soundscapes/
- JavaScript feature extraction library: https://meyda.js.org/
- JavaScript visualization library: https://d3js.org/
- Machine learning library: https://scikit-learn.org/stable/
- Deep learning library: https://www.tensorflow.org/
- Python data analysis library: https://pandas.pydata.org/
- Python visualization library: https://matplotlib.org/
