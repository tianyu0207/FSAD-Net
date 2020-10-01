import os
import torch.utils.data
from PIL import Image


class MyDataLoader(torch.utils.data.Dataset):

    # constructor of the class
    def __init__(self, normal_path, abnormal_path, test_path, transform=None, train=False, validate=False, test=False):
        assert (train is True and test is False and validate is False) or \
               (train is False and test is True and validate is False) or \
               (train is False and test is False and validate is True), \
            ">> Error: The mode in use is one of [train, validate, test]"

        if validate is True:
            assert abnormal_path is not None and normal_path is not None, \
                '>> ERROR: we need normal and abnormal path in validate mode'

        self.test = test
        self.train = train
        self.validate = validate
        self.transform = transform

        normal_images = []
        abnormal_images = []
        if normal_path is not None:
            normal_images = [os.path.join(normal_path, img) for img in os.listdir(normal_path)]
            normal_images = sorted(normal_images, key=lambda x: int(str(x).
                                                                    split('/')[-1].
                                                                    split('.')[0].
                                                                    split('_')[-1]))

        if abnormal_path is not None:
            abnormal_images = [os.path.join(abnormal_path, img) for img in os.listdir(abnormal_path)]
            abnormal_images = sorted(abnormal_images, key=lambda x: int(str(x).
                                                                        split('/')[-1].
                                                                        split('.')[0].
                                                                        split('_')[-1]))
        if self.train is True and abnormal_path is not None:
            print('>> Warning: There is no abnormal images will be used in training set!')

        images = None
        if self.train:
            # numbers of images in data set
            images_number = len(normal_images)
            images = normal_images[int(0 * images_number):] + abnormal_images

        if self.validate:
            images_number = len(normal_images)
            images = normal_images[:int(0 * images_number)] + abnormal_images

        # random.shuffle(images)
        self.images = images

    def __getitem__(self, index):
        image_path = self.images[index]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        labels = []
        if self.test:
            return
        else:
            label = "1_"+ str(image_path.split('/')[-1].split('_')[1].split('.')[0]) \
                if image_path.split('/')[-1].split('_')[0] == 'normal' \
                else "0_" + str(image_path.split('/')[-1].split('_')[1].split('.')[0])

        data = Image.open(image_path)
        # data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.images)
