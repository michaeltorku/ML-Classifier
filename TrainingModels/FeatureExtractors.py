#Principal Component Analysis (PCA)
import sklearn
from sklearn.decomposition import PCA

def _PCA(variance_percentage = 0.95, train_x, test_x):
    """ This method peforms principal component analysis on a given set of training data
        and transforms the test data to fit the analysis.
        
        @param: variance_percentage The percentage of variance the principal components should account for
        @param: train_x The training data for the PCA model 
        @param: test_x  The testing data the PCA model will transform

        @return: X_train The new features to train the model
    """
    PCA_Model = create_PCA(variance_percentage, train_x)
    


def create_PCA(variance_percentage = 0.95, train_x):
    """This method creates a principal component model for a given set of training data

        @param: variance_percentage The percentage of variance the principal components should account for
        @param: train_x The data to train the PCA model
        @return: model The trained PCA model
    """
    #Trains a PCA model to find features accounting for variance_percentage of the dataset train_x
    model = PCA(variance_percentage).fit(train_x)

    return model

def transform_PCA(pca_model, test_x):
    """This method transforms a given dataset using the inputted model

        @param: pca_model The pre-trained model that would transform the given dataset
        @param: test_x The dataset for the model to transform
        @return: features The extracted features
    """

    features = pca_model.transform(test_x)
    
    return features

