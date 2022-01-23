import sys
sys.path.append('/home/linakim') # append your path
from utils.dataset import CreateTrainDataSet, set_dict
from models.TF.Classifier.LeNet.Lenet import LeNet


def train():
    pass


if __name__ == "__main__":
    dict_list = set_dict()
    path = 'dataset'
    x_train, y_train = CreateTrainDataSet(path)

    print(x_train.shape, y_train.shape)
