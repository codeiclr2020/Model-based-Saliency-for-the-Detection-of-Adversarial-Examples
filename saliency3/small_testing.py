import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import h5py
from sal.small import SimpleClassifier
from sal.saliency_model import SaliencyModel, SaliencyLoss
NUM_CLASSES= 10

#0 -> cat
#in categorical, cat becomes the first column hence column 0 entry is 1
#therefoer if we take only the first column, then it becomes 1


def generate_cat_image(sal_func, X, Y):
    map = None
    max_val = -1
    map_not_found = True
    if Y == 1: #image is a cat
        check_val = [3]
    else:
        check_val = [5]
    if isinstance(X, np.ndarray):
        X = X[:, :32, :32]
        X = np.expand_dims(X, 0)
        X= torch.tensor(X)

    map = sal_f(X, torch.tensor(check_val))
    map = map[0].reshape([32, 32])

    max_val = np.amax(abs(map.detach().numpy())) / 1.0
    mean_val = np.mean(map.detach().numpy()) / 1.0
    return map, max_val, mean_val, 1.0


if __name__ == '__main__':

    # set directory
    directory = os.getcwd() + '/processed_data/'

    # get pretrained saliency model
    model = SimpleClassifier(base_channels=32)
    model.restore('SmallBlackBoxModel')

    # Default saliency model with pretrained resnet50 feature extractor, produces saliency maps which have resolution 4 times lower than the input image.
    sal_f = SaliencyModel(model, 3, 32, 3, 32, fix_encoder=True, use_simple_activation=False, allow_selector=True,
                          num_classes=NUM_CLASSES)

    #sal_f.minimialistic_restore('yoursaliencymodel')

    # mean color
    mean_color = 0

    # iterate over datasets
    for set in range(1):
        # load in data
        with h5py.File(directory +str(set), 'r') as f:
            X_train = f['train']['X'].value
            Y_train = f['train']['Y'].value

        #save stats and images
        temp_vals = []
        mean_value = []
        adv_image = []
        sal_ind =[]
        maps=[]

        for j in range(4):
            begin = j*500
            end= (j+1)*500
            if j==3:
                end=-1
            X_temp = X_train[begin:end,:,:,:]
            Y_temp = Y_train[begin:end]

            # switch axis as pytroch requires no. channels x height x width
            X_t= (np.moveaxis(X_temp, 3, 1))
            Y_t = Y_temp[:,0].astype(int)

            for k in xrange(len(X_temp)):
                # iterate over images
                map, max_val, mean_val, sal_ind = generate_cat_image(sal_f, X_t[k], Y_t[k])

                maps.append(map.detach().numpy())
                temp_vals.append(max_val)
                mean_value.append(mean_val)

                # generate image
                temp_image = X_temp[k].reshape([224, 224, 3])
                temp_image = temp_image[:32,:32,:]
                temp_map = np.expand_dims(map.detach().numpy(), -1)
                temp_map = np.repeat(temp_map, repeats = 3, axis= 2)

                image = temp_image*(1-temp_map) + temp_map*mean_color
                adv_image.append(image)

        plt.subplot(2, 1, 1)
        num_bins = 20
        plt.hist(temp_vals, num_bins, normed=1, facecolor='blue', alpha=0.5)
        plt.title('Maximum Values')
        plt.show()

        plt.subplot(2, 1, 2)
        plt.hist(mean_value, num_bins, normed=1, facecolor='blue', alpha=0.5)
        plt.title('Mean Values')

        plt.show()
        print(temp_vals)
        print(mean_value)

        with h5py.File(directory + 'A_Trial' + str(set), 'w') as f:
            tr = f.create_group('train')
            tr.create_dataset('X', data=X_train)
            tr.create_dataset('Y', data=Y_train)
            tr.create_dataset('SAL', data=adv_image)
            tr.create_dataset('SAL_indicator', data = sal_ind)
            tr.create_dataset('Map', data =maps)