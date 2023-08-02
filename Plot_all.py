import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np



def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name="roc.png"):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    fig = plt.figure()
    plt.plot(
        false_positive_rate, true_positive_rate, color='red', lw=2, label="ROC Curve (area = {:.4f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)
    plt.close()


def plot_accuracy_lfw(log_file,log_file2,log_file3,log_file4,log_file5,log_file6, epochs, figure_name="Verification_accuracies.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
        fig, ax = plt.subplots()
        ax.plot(epoch_list, accuracy_list, color='red', label='PointMLP')
        
    if log_file2 is not None:
        with open(log_file2, 'r') as f:
            lines = f.readlines()
            epoch_list = [int(line.split('\t')[0]) for line in lines]
            accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='blue', label='PointMLP-Elite')
            
    if log_file3 is not None:
        with open(log_file3, 'r') as f:
            lines = f.readlines()
            epoch_list = [int(line.split('\t')[0]) for line in lines]
            accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='orange', label='PointMLP-Elite8')
            
    if log_file4 is not None:
        with open(log_file4, 'r') as f:
            lines = f.readlines()
            epoch_list = [int(line.split('\t')[0]) for line in lines]
            accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='purple', label='PointNet++SSG')
            
    if log_file5 is not None:
        with open(log_file5, 'r') as f:
            lines = f.readlines()
            epoch_list = [int(line.split('\t')[0]) for line in lines]
            accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='cyan', label='PointNet++SSG-Elite')
            
    if log_file6 is not None:
        with open(log_file6, 'r') as f:
            lines = f.readlines()
            epoch_list = [int(line.split('\t')[0]) for line in lines]
            accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='gray', label='PointNet')
            
    plt.yticks(np.arange(0.00, 1.05, 0.1))
    plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracies Plot')
    plt.legend(loc='lower right')
    plt.savefig(figure_name, dpi=300)
    plt.show()

def plot_accuracy_embeddings(log_file,log_file2, epochs, figure_name="Embeddings_accuracies.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
        fig, ax = plt.subplots()
        ax.plot(epoch_list, accuracy_list, color='red', label='(1x4096) embedding')
        
    if log_file2 is not None:
        with open(log_file2, 'r') as f:
            lines = f.readlines()
            epoch_list = [int(line.split('\t')[0]) for line in lines]
            accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='blue', label=' (1x512) embedding')
            
            
    plt.yticks(np.arange(0.00, 1.05, 0.1))
    plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy on PointMLP-Elite')
    plt.title('Validation Accuracies on PointMLP-Elite Plot')
    plt.legend(loc='lower right')
    plt.savefig(figure_name, dpi=300)
    plt.show()
    
def plot_valid_triplets(log_1,log_2,log_3,log_4,log_5,log_6, epochs, figure_name="valid_triplets.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_1, 'r') as f:
        next(f)
        lines = f.readlines()
        epoch_list = [int(float(line.split('\t')[0])) for line in lines]
        valid_list = [round(float(line.split('\t')[3]), 2) for line in lines]
        fig, ax = plt.subplots()
        ax.plot(epoch_list, valid_list, color='red', label='PointMLP')
        
    if log_2 is not None:
        with open(log_2, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            valid_list = [round(float(line.split('\t')[3]), 2) for line in lines]
            ax.plot(epoch_list, valid_list, color='blue', label='PointMLP-Elite')
            
    if log_3 is not None:
        with open(log_3, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            valid_list = [round(float(line.split('\t')[3]), 2) for line in lines]
            ax.plot(epoch_list, valid_list, color='orange', label='PointMLP-Elite8')
            
    if log_4 is not None:
        with open(log_4, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            accuracy_list = [round(float(line.split('\t')[3]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='purple', label='PointNet++SSG')
            
    if log_5 is not None:
        with open(log_5, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            accuracy_list = [round(float(line.split('\t')[3]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='cyan', label='PointNet++SSG-Elite')
            
    if log_6 is not None:
        with open(log_6, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            accuracy_list = [round(float(line.split('\t')[3]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='gray', label='PointNet')
            
    plt.yticks(np.arange(0, 3400, 200))
    plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Valid Triplets')
    plt.title('Valid Triplets Plot')
    plt.legend(loc='upper right')
    plt.savefig(figure_name, dpi=300)
    plt.show()
    
def plot_loss_triplets(log_1,log_2,log_3,log_4,log_5,log_6, epochs, figure_name="loss_triplets.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_1, 'r') as f:
        next(f)
        lines = f.readlines()
        epoch_list = [int(float(line.split('\t')[0])) for line in lines]
        valid_list = [round(float(line.split('\t')[2]), 2) for line in lines]
        fig, ax = plt.subplots()
        ax.plot(epoch_list, valid_list, color='red', label='PointMLP')
        
    if log_2 is not None:
        with open(log_2, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            valid_list = [round(float(line.split('\t')[2]), 2) for line in lines]
            ax.plot(epoch_list, valid_list, color='blue', label='PointMLP-Elite')
            
    if log_3 is not None:
        with open(log_3, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            valid_list = [round(float(line.split('\t')[2]), 2) for line in lines]
            ax.plot(epoch_list, valid_list, color='orange', label='PointMLP-Elite8')
            
    if log_4 is not None:
        with open(log_4, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            accuracy_list = [round(float(line.split('\t')[2]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='purple', label='PointNet++SSG')
            
    if log_5 is not None:
        with open(log_5, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            accuracy_list = [round(float(line.split('\t')[2]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='cyan', label='PointNet++SSG-Elite')
            
    if log_6 is not None:
        with open(log_6, 'r') as f:
            next(f)
            lines = f.readlines()
            epoch_list = [int(float(line.split('\t')[0])) for line in lines]
            accuracy_list = [round(float(line.split('\t')[2]), 2) for line in lines]
            ax.plot(epoch_list, accuracy_list, color='gray', label='PointNet')
            
    plt.yticks(np.arange(0, 0.55, 0.05))
    plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss ')
    plt.title('Train Loss vs Epoch Plot')
    plt.legend(loc='lower left')
    plt.savefig(figure_name, dpi=300)
    plt.show()

def plot_learning_rates(log_1, epochs, figure_name="learning_rates.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_1, 'r') as f:
        next(f)
        lines = f.readlines()
        epoch_list = [int(float(line.split('\t')[0])) for line in lines]
        learning_rate_list = [(float(line.split('\t')[1])) for line in lines]
        fig, ax = plt.subplots()
        ax.plot(epoch_list, learning_rate_list, color='red')
            
    plt.yticks(np.arange(0.0001,0.0105,0.0005))
    plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epoch Plot')
    plt.savefig(figure_name, dpi=300)
    plt.show()
    
def main():
    log_file = "./checkpoints/pointMLPTriplet-20230311014131-6494/logs/Bosphorus_pointMLPTriplet_log_triplet_seed_6494.txt"
    log_file2 = "./checkpoints/pointMLPEliteTriplet-20230310130117-9721/logs/Bosphorus_pointMLPEliteTriplet_log_triplet_seed_9721.txt"
    log_file3 = "./checkpoints/pointMLPElite8Triplet-20230310193745-3600/logs/Bosphorus_pointMLPElite8Triplet_log_triplet_seed_3600.txt"
    log_file4 = "./checkpoints/pointnet2ssgTriplet-20230309190429-8728/logs/Bosphorus_pointnet2ssgTriplet_log_triplet_seed_8728.txt"
    log_file5 = "./checkpoints/pointnet2ssgeliteTriplet-20230313175411-6822/logs/Bosphorus_pointnet2ssgeliteTriplet_log_triplet_seed_6822.txt"
    log_file6 = "./checkpoints/pointnetTriplet_v2-20230313034039-901/logs/Bosphorus_pointnetTriplet_v2_log_triplet_seed_901.txt"
    epochs = 49
    log_1 = "./checkpoints/pointMLPTriplet-20230311014131-6494/log.txt"
    
    log_2 = "./checkpoints/pointMLPEliteTriplet-20230310130117-9721/log.txt"
    log_3 = "./checkpoints/pointMLPElite8Triplet-20230310193745-3600/log.txt"
    log_4 = "./checkpoints/pointnet2ssgTriplet-20230309190429-8728/log.txt"
    log_5 = "./checkpoints/pointnet2ssgeliteTriplet-20230313175411-6822/log.txt"
    log_6 = "./checkpoints/pointnetTriplet_v2-20230313034039-901/log.txt"
    
    log_file_01 = "./checkpoints/pointMLPEliteTriplet-20230328203019-445/logs/Bosphorus_pointMLPEliteTriplet_log_triplet_seed_445.txt"
    
    
    #plot_accuracy_lfw(log_file,log_file2,log_file3,log_file4,log_file5,log_file6, epochs, figure_name="Validation_accuracies.png")
    plot_accuracy_embeddings(log_file2,log_file_01, epochs, figure_name="embedding accuracy.png")
    #plot_valid_triplets(log_1,log_2,log_3,log_4,log_5,log_6, epochs, figure_name="Valid_triplets.png")
    #plot_loss_triplets(log_1,log_2,log_3,log_4,log_5,log_6, epochs, figure_name="Loss_triplets.png")
    #plot_learning_rates(log_1, epochs, figure_name="learning_rates.png")
if __name__ == '__main__':
    main()