Hii Everyone,

I am currently undertaking the development of a speech emotion detection classifier. However, prior to delving into the project, it is essential to familiarize ourselves with the concept of speech recognition (SER) and the underlying motivations for this endeavor. SER, which stands for Speech Emotion Recognition, involves the endeavor to discern human emotions and affective states from speech. This approach capitalizes on the notion that vocal characteristics such as tone and pitch often convey underlying emotions. Animals such as dogs and horses also rely on this phenomenon to comprehend human emotions.

The necessity for SER has been steadily increasing, as emotion recognition constitutes a significant aspect of speech recognition and has gained substantial popularity. While machine learning techniques can be employed for emotion recognition, this project aims to leverage deep learning methods to discern emotions from data.

The utilization of SER extends to various domains, including call centers, where it enables the classification of calls based on emotions, thus serving as a performance metric for conversational analysis. This application allows companies to identify unsatisfied customers, evaluate customer satisfaction levels, and improve their services accordingly.

Moreover, SER can be integrated into in-car systems, where information regarding the driver's mental state can be utilized to enhance safety measures and prevent accidents from occurring.

The datasets employed in this project encompass the following:

1. Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)
2. Surrey Audio-Visual Expressed Emotion (Savee)

By leveraging these datasets, we aim to achieve accurate and robust emotion recognition through our speech emotion detection classifier.

Development of this project require following steps:
1.	Data Preparation
   
      •	As we are working with four different datasets, so i will be creating a dataframe storing all emotions of the data in dataframe with their paths.
      •	We will use this dataframe to extract features for our model training.
    Understanding the datasets:
    
    1. Ravdess Dataframe
    
        The RAVDESS dataset, known as the Ryerson Audio-Visual Database of Emotional Speech and Song, comprises audio files specifically focused on speech. These files are in 16-bit, 48kHz .wav format and can be obtained in their entirety, including both speech and song, along with audio and video components, from Zenodo. The construction and perceptual validation of the RAVDESS dataset are extensively detailed in an Open Access paper published in PLoS ONE.
      	
        For the section currently being described, it contains a total of 1440 files, corresponding to 60 trials per actor and 24 actors, resulting in the calculation of 1440 files. The dataset features 24 professional actors, equally split between 12 females and 12 males, who vocalize two statements matched for lexical content, using a neutral North American accent. The speech emotions conveyed in the dataset include calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is portrayed at two levels of emotional intensity: normal and strong, with the addition of a neutral expression.
    
      	The filenames assigned to the 1440 files adhere to a specific convention, comprising a seven-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers serve to define the characteristics of the stimuli:
      
      - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
      - Vocal channel (01 = speech, 02 = song)
      - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
      - Emotional intensity (01 = normal, 02 = strong) *Note: 'Neutral' emotion does not have a strong intensity.
      - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
      - Repetition (01 = 1st repetition, 02 = 2nd repetition)
      - Actor (01 to 24; odd-numbered actors are male, even-numbered actors are female)
    
    For instance, in the filename example "03-01-06-01-02-01-12.wav," the breakdown is as follows:
    
    - Audio-only (03)
    - Speech (01)
    - Fearful (06)
    - Normal intensity (01)
    - Statement "dogs" (02)
    - 1st Repetition (01)
    - 12th Actor (12)
    - Female, as the actor ID number is even.
    
    This comprehensive dataset provides valuable resources for emotion recognition research, particularly in the realm of speech analysis.

    2. Surrey Audio-Visual Expressed Emotion (SAVEE)
       
    In my pursuit of developing an emotion classifier from audio data, I fortuitously came across the SAVEE dataset, which constitutes one of the four crucial datasets for my project. Notably, this dataset exclusively comprises recordings from male speakers and boasts exceptionally high-quality audio. However, due to the inherent gender imbalance in the dataset, it is advisable to supplement it with additional datasets featuring more female speakers.
              
    The SAVEE database features recordings from four native English male speakers, namely DC, JE, JK, and KL, who are postgraduate students and researchers at the University of Surrey, aged between 27 and 31 years. Emotions have been psychologically classified into discrete categories, including anger, disgust, fear, happiness, sadness, surprise, and a neutral category. The dataset includes 15 TIMIT sentences per emotion: three common sentences, two emotion-specific sentences, and ten generic sentences, each phonetically balanced and unique for every emotion. Additionally, to provide recordings for the neutral category, the three common and two sets of six emotion-specific sentences were recorded neutrally, resulting in a total of 120 utterances per speaker.
    
    I would like to express my gratitude to the University of Surrey for curating this exceptional dataset, which has greatly contributed to my research endeavors.
      
2.	Data Augmentation
    •	Data augmentation is the process by which we create new synthetic data samples by adding small perturbations on our initial training set.
    •	To generate syntactic data for audio, we can apply noise injection, shifting time, changing pitch and speed.
    •	The objective is to make our model invariant to those perturbations and enhace its ability to generalize.
    •	In order to this to work adding the perturbations must conserve the same label as the original training sample.
    •	In images data augmention can be performed by shifting the image, zooming, rotating ...
    
3.	Feature Extraction
     Extraction of features is a very important part in analyzing and finding relations between different things. As we already know that the data provided of audio cannot be understood by the models directly so we need to convert them into an understandable format for which feature extraction is used.
      The audio signal is a three-dimensional signal in which three axes represent time, amplitude     and frequency.
      I am no expert on audio signals and feature extraction on audio files so i need to search and found a very good blog written by Askash Mallik on feature extraction.
  
      As stated there with the help of the sample rate and the sample data, one can perform several transformations on it to extract valuable features out of it.
  
      1.	Zero Crossing Rate : The rate of sign-changes of the signal during the duration of a particular frame.
      2.	Energy : The sum of squares of the signal values, normalized by the respective frame length.
      3.	Entropy of Energy : The entropy of sub-frames’ normalized energies. It can be interpreted as a measure of abrupt changes.
      4.	Spectral Centroid : The center of gravity of the spectrum.
      5.	Spectral Spread : The second central moment of the spectrum.
      6.	Spectral Entropy : Entropy of the normalized spectral energies for a set of sub-frames.
      7.	Spectral Flux : The squared difference between the normalized magnitudes of the spectra of the two successive frames.
      8.	Spectral Rolloff : The frequency below which 90% of the magnitude distribution of the spectrum is concentrated.
      9.	MFCCs Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.
      10.	Chroma Vector : A 12-element representation of the spectral energy where the bins represent the 12 equal-tempered pitch classes of western-type music (semitone spacing).
      11.	Chroma Deviation : The standard deviation of the 12 chroma coefficients.
      In this project i am not going deep in feature selection process to check which features are good for our dataset rather i am only extracting 5 features:
      •	Zero Crossing Rate
      •	Chroma_stft
      •	MFCC
      •	RMS(root mean square) value
      •	MelSpectogram to train our model.
4.	Data Preparation
    •	As of now we have extracted the data, now we need to normalize and split our data for training and testing.
5.	Modelling
    •	Use appropriate model for training.
6.	Testing

To download dataset:
1. RAVDESS: https://drive.google.com/file/d/186_PxaxUIddi8rK5TzLhopx8Y_bDMSWb/view?usp=sharing
2. SAVEE: https://drive.google.com/file/d/185ii0Nw4g7l4zhYXwVeNbboK1WDePGWS/view?usp=sharing


