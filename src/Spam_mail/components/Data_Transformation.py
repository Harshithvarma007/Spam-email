import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from Spam_mail import logger
from Spam_mail.config.configuration import *

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def Data_transformation(self):
        # Read the data from CSV using the path from config
        df = pd.read_csv(self.config.data_path)
        logger.info('Reading Data...')
        logger.info('Vectorization initiated')
        # Vectorization using CountVectorizer
        cv = CountVectorizer()
        x = df['Message']
        y = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)  # Convert labels to binary values
        x_vectorized = cv.fit_transform(x)
        logger.info('Vectorization Completed')


        # Save CountVectorizer to a file
        vectorizer_path = os.path.join(self.config.root_dir, 'count_vectorizer.pkl')
        joblib.dump(cv, vectorizer_path)

        # SMOTE oversampling
        smote = SMOTE(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(x_vectorized, y)
        logger.info('Oversampling Completed')
        # Splitting Dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)
        logger.info('Data Splitting is Completed')
        return x_train, x_test, y_train, y_test
