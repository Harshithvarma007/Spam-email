import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from Spam_mail import logger
import tensorflow as tf
class ModelEvaluation:
    def __init__(self, config):
        self.config = config
    
    def eval(self, x_test, y_test):
        # Load the model from the specified path
        logger.info('Model Loading')
        model = load_model(self.config.model_path)
        logger.info('Model Loaded')        
        # Evaluate the model

        x_test_sparse = tf.sparse.from_dense(x_test)

    # Use x_test_sparse in model prediction
        y_pred_prob = model.predict(x_test_sparse)
        # y_pred_prob = model.predict(x_test)
        logger.info('Prediction is completed')
        y_pred = (y_pred_prob > 0.5).astype(int)
        # Reorder sparse indices if necessary
        x_test_sparse = tf.sparse.reorder(x_test)        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 score: {f1:.4f}')
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plotting Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="coolwarm", fmt="d")
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()
