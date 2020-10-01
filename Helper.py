from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy
import os
import torchvision
from sklearn.metrics import roc_curve, auc


def poly_lr_scheduler(my_optimizer, init_lr, epoch,
                      lr_decay_iter=1,
                      max_iter=100,
                      power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param epoch is a current epoch
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
        :param my_optimizer is optimizer

    """
    if epoch % lr_decay_iter or epoch > max_iter:
        return my_optimizer

    lr = init_lr * (1 - epoch / max_iter) ** power
    for param_group in my_optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mnist_plot_encoded_3d_chart(number_value, encoded_data):
    fig = plt.figure(2)
    ax = Axes3D(fig)
    x_axis, y_axis, z_axis = encoded_data.data[:, 0].cpu().detach().numpy(), \
                             encoded_data.data[:, 1].cpu().detach().numpy(), \
                             encoded_data.data[:, 2].cpu().detach().numpy()

    for x, y, z, s in zip(x_axis, y_axis, z_axis, number_value):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(x_axis.min(), x_axis.max())
    ax.set_ylim(y_axis.min(), y_axis.max())
    ax.set_zlim(z_axis.min(), z_axis.max())
    plt.show()


def mnist_get_data_set(dir_path, get_number=None, not_number=None, train=True):
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((31, 31)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_set = torchvision.datasets.MNIST(dir_path, train=train, transform=img_transform)
    idx = 0
    if get_number is not None or not_number is not None:
        if get_number is not None:
            idx = data_set.train_labels == get_number
        elif not_number is not None:
            idx = data_set.train_labels != not_number
        data_set.train_labels = data_set.train_labels[idx]
        data_set.train_data = data_set.train_data[idx]
    return data_set


def plot_2d_chart(x1, y1, label1='predict_results',
                  x2=None, y2=None, label2=None, save_path=None, title=None):
    plt.plot(x1, y1, color='red', label=label1, marker='o', mec='r', mfc='w')
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2, color='green', label=label2, marker='*', mec='g', mfc='w')
        # plt.plot(x2, y2, color='green', label=label2,
        #          marker='o',
        #          mec='green',
        #          mfc='w')
    plt.xlabel('labels')
    plt.xticks(rotation=45)
    plt.ylabel('loss')
    plt.legend()
    if title is not None:
        plt.title(title)
    else:
        plt.title('line_chart_for_anomaly_detector')
    plt.grid()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def view_images(title_list, image_list, task_title=None, size=5, axis=False):
    row = numpy.int(numpy.ceil(len(image_list) / 2))
    column = numpy.int(numpy.ceil(len(image_list) / row))
    plt.figure(task_title)
    plt.figure(task_title, figsize=(size, size))
    for col_index in range(column):
        for row_index in range(row):
            num = col_index * row + row_index
            if num == len(image_list) or num == len(title_list):
                break
            plt.subplot(row, column, num + 1)
            plt.title(title_list[num])
            if not axis:
                plt.axis('off')
            plt.imshow(image_list[num])

    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


def plot_abnormal_normal_chart(abnormal, normal,
                               save=None, show=False):
    plt.plot(numpy.arange(0, 128), abnormal, color='red', label='abnormal_only_zeros ',
             marker='o',
             mec='r', mfc='w')

    plt.plot(numpy.arange(128, 256), normal, color='green', label='normal_no_zeros',
             marker='o',
             mec='green',
             mfc='w')
    plt.xlabel('labels')
    plt.xticks(rotation=45)
    plt.ylabel('loss')
    plt.legend()
    plt.title('line_chart_for_anomaly_detector')
    plt.grid()
    if show:
        plt.show()
    if save is not None:
        dir_path = str(save).split('/')[0]
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        plt.savefig('{}.png'.format(save))
    plt.close()


def mnist_get_visualize_data(input_image, output_image):
    pic_total_1 = numpy.array(input_image[0])
    for i in range(input_image.shape[0] - 1):
        pic_total_1 = numpy.concatenate((pic_total_1, input_image[i + 1]), axis=1)

    pic_total_2 = numpy.array(output_image[0])
    for i in range(output_image.shape[0] - 1):
        pic_total_2 = numpy.concatenate((pic_total_2, output_image[i + 1]), axis=1)
    pic_total = numpy.concatenate((pic_total_1, pic_total_2), axis=0)
    return pic_total


def draw_roc(tp_list, fp_list, title=None, save_path = None):
    # auc drawing
    auc_curve = auc(fp_list, tp_list)

    plt.plot(fp_list, tp_list, color='red', label='AUC area:(%0.5f)' % auc_curve)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if title is not None:
        plt.title('Roc_Curve based on {}'.format(title))
    plt.legend(loc='lower right')
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


