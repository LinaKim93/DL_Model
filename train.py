from models.TF.LeNet.Lenet import LeNet
from utils.CallData import GetTrainData, set_dict


def train():
    pass


if __name__ == "__main__":
    # dict_list = set_dict()
    path = 'dataset'
    x_train, y_train = GetTrainData(path)

    print(x_train.shape, y_train.shape)
