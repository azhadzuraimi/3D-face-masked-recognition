"""
Usage:
python main.py --model PointMLP --msg demo
"""

import argparse
import os
import logging
import datetime
# import torch
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.utils.data
# import torch.utils.data.distributed
# from torch.utils.data import DataLoader
# import models as models
from utils import Logger, save_args ,mkdir_p
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from data.TestBosphorus import TestBosphorus
#from datasets.TripletLossDataset import TripletFaceDataset
from data.custom_dataset_v3 import TripletPoints_Dataset
from losses.triplet_loss import TripletLoss, TripletLossPointnet
from validate_on_Bosphorus import evaluate_bd415
from plot import plot_roc_lfw, plot_accuracy_lfw
from tqdm import tqdm
from models.pointmlp import pointMLPEliteTriplet,pointMLPElite8Triplet, pointMLPTriplet
from models.Pointnet2 import pointnet2ssgeliteTriplet, pointnet2ssgTriplet
from models.Pointnet import pointnetTriplet, pointnetTriplet_v2
from torch.optim.lr_scheduler import CosineAnnealingLR

#to resume make sure all argument is same as previous approach except checkpoint, resume_path and  

parser = argparse.ArgumentParser(description="Training a MLP facial recognition model using Triplet Loss.")
parser.add_argument('--dataroot', '-d', type=str, 
                    help="(REQUIRED) Absolute path to the training dataset folder"
                    )
parser.add_argument('--validate_data_path', type=str, default="/mnt/h/Work/code/Master-project/Python_recognition/classification_Bosphorus/data/test_file",
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--training_dataset_csv_path', type=str, default='/mnt/h/Work/code/Master-project/Python_recognition/classification_Bosphorus/data/BosphorusCustom.csv',
                    help="Path to the csv file containing the image paths of the training dataset"
                    )
parser.add_argument('-c', '--checkpoint', type=str, default="./checkpoints/pointMLPEliteTriplet-20230328203019-445", metavar='PATH',
                        help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume_path', default='pointMLPEliteTriplet_-20230328203019_18.pt',  type=str,
                    help='path to latest model checkpoint: (pointMLPTriplet_-20230311014131_35.pt file) (default: None)'
                    )
parser.add_argument('--msg', type=str, help='message after checkpoint')
parser.add_argument('--epochs', default=50, type=int,
                    help="Required training epochs (default: 150)"
                    )

parser.add_argument('--seed', type=int , help='random seed')
parser.add_argument('--iterations_per_epoch', default=100, type=int,
                    help="Number of training iterations per epoch (default: 5000)"
                    )
parser.add_argument('--model_architecture', type=str, default="pointMLPEliteTriplet", choices=["pointMLPTriplet","pointMLPEliteTriplet","pointMLPElite8Triplet"
                                                                                          "pointnet2ssgTriplet","pointnet2ssgeliteTriplet","pointnetTriplet_v2",
                                                                                          "pointnetTriplet"],
                    help="The required model architecture for training: ('pointMLPEliteTriplet'), (default: 'pointMLPElite8Triplet')"
                    )
parser.add_argument('--normalized_out', default=True, type=bool,
                    help="normalzied emdbeddings"
                    )
parser.add_argument('--embedding_dimension', default=512, type=int,
                    help="Dimension of the embedding vector (default: 4096)"
                    )
parser.add_argument('--num_human_identities_per_batch', default=32, type=int,
                    help="Number of set human identities per generated triplets batch. (Default: 32)."
                    )
parser.add_argument('--batch_size', default=32, type=int,
                    help="Batch size (default: 32)"
                    )
parser.add_argument('--batch_accum', default=4, type=int,
                    help="gradient accumulate."
                    )
parser.add_argument('--test_batch_size', default=32, type=int,
                    help="Batch size for Bosphorus dataset (2000 pairs) (default: 32)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 4)"
                    )
parser.add_argument('--optimizer', type=str, default="adam", choices=["sgd", "adagrad", "rmsprop", "adam"],
                    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adagrad')"
                    )
parser.add_argument('--learning_rate', default=0.01, type=float,
                    help="Learning rate for the optimizer (default: 0.075)"
                    )
parser.add_argument('--min_lr', default=0.0001, type=float,
                    help="Learning rate for the optimizer (default: 0.005)"
                    )
parser.add_argument('--margin', default=0.5, type=float,
                    help='margin for triplet loss (default: 0.5)'
                    )
parser.add_argument('--use_semihard_negatives', default=False, type=bool,
                    help="If True: use semihard negative triplet selection. Else: use hard negative triplet selection (Default: False)"
                    )
parser.add_argument('--training_triplets_path', default=None, type=str,
                    help="Path to training triplets numpy file in 'datasets/generated_triplets' folder to skip training triplet generation step for the first epoch."
                    )
parser.add_argument('--numpts', default=2048, type=int,
                    help="Path to training triplets numpy file in 'datasets/generated_triplets' folder to skip training triplet generation step for the first epoch."
                    )

args = parser.parse_args()


def set_model_architecture(model_architecture, numpts, embedding_dimension, normalized_out):
    if model_architecture == "pointMLPTriplet":
        model = pointMLPTriplet(
            out_embedding = embedding_dimension,
            numpts = numpts,
            embedding_normalize = normalized_out
        )
    if model_architecture == "pointMLPEliteTriplet":
        model = pointMLPEliteTriplet(
            out_embedding = embedding_dimension,
            numpts = numpts,
            embedding_normalize = normalized_out
        )
    if model_architecture == "pointMLPElite8Triplet":
        model = pointMLPElite8Triplet(
            out_embedding = embedding_dimension,
            numpts = numpts,
            embedding_normalize = normalized_out
        )
    if model_architecture == "pointnet2ssgTriplet":
        model = pointnet2ssgTriplet(
            out_embedding = embedding_dimension,
            embedding_normalize = normalized_out
        )      
    if model_architecture == "pointnet2ssgeliteTriplet":
        model = pointnet2ssgeliteTriplet(
            out_embedding = embedding_dimension,
            embedding_normalize = normalized_out
        )
    if model_architecture == "pointnetTriplet":
        model = pointnetTriplet(
            out_embedding = embedding_dimension,
            embedding_normalize = normalized_out
        )    
    if model_architecture == "pointnetTriplet_v2":
        model = pointnetTriplet_v2(
            out_embedding = embedding_dimension,
            embedding_normalize = normalized_out
        )  
   
        
    print("Using {} model architecture..".format(model_architecture))
    if normalized_out:
        print("with normalized embedding output")
    else:
        print("without normalized embedding output")
    return model

def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        device='cuda'
        model = model.to(device)
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        device='cuda' 
        model = model.to(device)
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu, device

def set_optimizer(optimizer, model, learning_rate):
    if optimizer == "sgd":
        optimizer_model = optim.SGD(
            params=model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            dampening=0,
            nesterov=False,
            weight_decay=1e-5
        )

    elif optimizer == "adagrad":
        optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=learning_rate,
            lr_decay=0,
            initial_accumulator_value=0.1,
            eps=1e-10,
            weight_decay=1e-5
        )

    elif optimizer == "rmsprop":
        optimizer_model = optim.RMSprop(
            params=model.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-08,
            momentum=0,
            centered=False,
            weight_decay=1e-5
        )

    elif optimizer == "adam":
        optimizer_model = optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=1e-5
        )

    return optimizer_model

def validate_Bosphorus(model, validate_dataloader, model_architecture, epoch , device, screen_logger):
    def printf(str):
        screen_logger.info(str)
        print(str)
    time_cost = datetime.datetime.now()
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        printf("Validating on Bosphorus+D415! ...")
        progress_bar = enumerate(tqdm(validate_dataloader))
        
        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            data_a = data_a.permute(0, 2, 1)
            data_b = data_b.permute(0, 2, 1)
            if model_architecture == "pointnetTriplett":
                output_a, _, _ = model(data_a)
                output_b, _, _ = model(data_b)
            else:
                output_a, output_b = model(data_a), model(data_b)
                
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance
            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_bd415(
            distances=distances,
            labels=labels,
            far_target=1e-1
        )
        time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
        # Print statistics and add to log
        printf("Accuracy on Bosphorus+D415: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}\tTime(s):{}".format(
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar),
                    np.std(tar),
                    np.mean(far),
                    time_cost
                )
        )
        save_logs_valid = os.path.join(args.checkpoint,'logs')
        if not os.path.isdir(save_logs_valid):
            mkdir_p(save_logs_valid)
        save_logs_valid = os.path.join(save_logs_valid,'Bosphorus_{}_log_triplet_seed_{}.txt'.format(model_architecture,args.seed))
        with open(save_logs_valid, 'a') as f:
            val_list = [
                epoch,
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar),
                time_cost
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

    try:
        if not os.path.isdir(os.path.join(args.checkpoint,"plots")):
            mkdir_p(os.path.join(args.checkpoint,"plots"))
        if not os.path.isdir(os.path.join(args.checkpoint,"plots/roc_plots")):
            mkdir_p(os.path.join(args.checkpoint,"plots/roc_plots"))
        if not os.path.isdir(os.path.join(args.checkpoint,"plots/accuracies_plots")):
            mkdir_p(os.path.join(args.checkpoint,"plots/accuracies_plots"))
        fig_name_ROC = os.path.join(args.checkpoint,"plots/roc_plots/roc_{}_seed_{}_epoch_{}_triplet.png".format(model_architecture,args.seed, epoch))
        fig_name_ACC = os.path.join(args.checkpoint,"plots/accuracies_plots/Bosphorus_accuracies_{}_seed_{}_epoch_{}_triplet.png".format(model_architecture,args.seed,epoch))
        # Plot ROC curve
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name=fig_name_ROC
        )
        # Plot LFW accuracies plot
        plot_accuracy_lfw(
            log_file=save_logs_valid,
            epochs=epoch,
            figure_name=fig_name_ACC
        )
    except Exception as e:
        print(e)

    return best_distances

def forward_pass(data, model, batch_size, accum_iter ,device):
    data = data.to(device)
    data = data.permute(0, 2, 1)
    embeddings = model(data)
    
    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:int(batch_size/accum_iter)]
    pos_embeddings = embeddings[int(batch_size/accum_iter): int(batch_size/accum_iter * 2)]
    neg_embeddings = embeddings[int(batch_size/accum_iter * 2):]

    return anc_embeddings, pos_embeddings, neg_embeddings, model

def forward_pass_Pointnet(data, model, batch_size, accum_iter ,device):
    data = data.to(device)
    data = data.permute(0, 2, 1)
    embeddings, m3x3, m64x64 = model(data)
    
    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:int(batch_size/accum_iter)]
    pos_embeddings = embeddings[int(batch_size/accum_iter): int(batch_size/accum_iter * 2)]
    neg_embeddings = embeddings[int(batch_size/accum_iter * 2):]

    return anc_embeddings, pos_embeddings, neg_embeddings, m3x3, m64x64, model

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataroot = os.path.join(BASE_DIR,"data","train_file") 
    # dataroot = args.dataroot
    if args.checkpoint is None or args.checkpoint is "":
        args.seed = np.random.randint(1, 10000)
    else:
        seed = args.checkpoint
        args.seed = int(seed.split("-")[2])
    validate_dataroot = args.validate_data_path
    training_dataset_csv_path = args.training_dataset_csv_path
    epochs = args.epochs
    iterations_per_epoch = args.iterations_per_epoch
    model_architecture = args.model_architecture
    embedding_dimension = args.embedding_dimension
    num_human_identities_per_batch = args.num_human_identities_per_batch
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    resume_path = args.resume_path
    num_workers = args.num_workers
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    min_lr = args.min_lr
    margin = args.margin
    use_semihard_negatives = args.use_semihard_negatives
    training_triplets_path = args.training_triplets_path
    flag_training_triplets_path = False
    start_epoch = 0
    numpts = args.numpts
    batch_accum = args.batch_accum

    
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg 
    if args.checkpoint is "" or args.checkpoint is None:
        args.checkpoint = 'checkpoints/' + args.model_architecture + message + '-' + str(args.seed) 
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)
    
    def printf(str):
        screen_logger.info(str)
        print(str)
        
    printf(f"args: {args}")
    
    if training_triplets_path is not None:
        printf("Load triplets file for the first training epoch")
        flag_training_triplets_path = True  
        
    printf('==> Preparing data for validation')   
    Bosphorus_dataloader = torch.utils.data.DataLoader(
        dataset=TestBosphorus(
            dir=validate_dataroot,
            pairs_path='/mnt/h/Work/code/Master-project/Python_recognition/classification_Bosphorus/data/Bosphorus_pair.txt',
        ),
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    printf('==> Building model..')
    model = set_model_architecture(
        model_architecture=model_architecture,
        numpts=numpts,
        embedding_dimension=embedding_dimension,
        normalized_out= args.normalized_out
    )

    printf('==> Load model to GPU or multiple GPUs if available..')
    model, flag_train_multi_gpu, device = set_model_gpu_mode(model)
    
    printf('==> Set seed..')
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        
    printf('==> Set Optimizer..')
    optimizer_model = set_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate
    )
    
    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(os.path.join(args.checkpoint,resume_path)):
            printf("Loading checkpoint {} ...".format(os.path.join(args.checkpoint,resume_path)))
            checkpoint = torch.load(os.path.join(args.checkpoint,resume_path))
            start_epoch = checkpoint['epoch'] + 1
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="Bosphorus+D415" + args.model_architecture, resume=True)
            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            printf("Checkpoint loaded: start epoch from checkpoint = {}".format(start_epoch))
        else:
            printf("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))
    else:
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="Bosphorus+D415" + args.model_architecture)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-valid-triplets', 'Best-distances-mean'])

    if use_semihard_negatives:
        printf("Using Semi-Hard negative triplet selection!")
    else:
        printf("Using Hard negative triplet selection!")

    start_epoch = start_epoch
    scheduler = CosineAnnealingLR(optimizer_model , epochs, eta_min=min_lr, last_epoch=start_epoch - 1)
    printf("Training using triplet loss starting for {} epochs:\n".format(epochs - start_epoch))
    num_triplets = iterations_per_epoch * batch_size
    for epoch in range(start_epoch, epochs):
        time_cost = datetime.datetime.now()
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, epochs, optimizer_model.param_groups[0]['lr']))
        running_loss = 0 #calculate loss 
        accuracy_triplet = 0 # triplet selection loss 100 mean
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(p=2)
        _training_triplets_path = None

        if flag_training_triplets_path:
            _training_triplets_path = training_triplets_path
            flag_training_triplets_path = False  # Only load triplets file for the first epoch

        # Re-instantiate training dataloader to generate a triplet list for this training epoch
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletPoints_Dataset(
                root_dir=dataroot,
                training_dataset_csv_path=training_dataset_csv_path,
                num_triplets=num_triplets,
                num_human_identities_per_batch=num_human_identities_per_batch,
                triplet_batch_size=batch_size,
                epoch=epoch,
                training_triplets_path=_training_triplets_path
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
        )
    
        # Training pass
        model.train()
        progress_bar = enumerate(tqdm(train_dataloader))

        for batch_idx, (batch_sample) in progress_bar:
            # Forward pass - compute embeddings
            anc_imgs = batch_sample['anc_img']
            pos_imgs = batch_sample['pos_img']
            neg_imgs = batch_sample['neg_img']
            
            for i in range (batch_accum):
                # Concatenate the input images into one tensor because doing multiple forward passes would create
                #  weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
                #  issues, batch accumulation for gradient accumulation implementation added
                from_slice = i*(batch_size/batch_accum)
                to_slice = (i+1)*(batch_size/batch_accum)
                anc_img = anc_imgs[int(from_slice): int(to_slice)]
                pos_img = pos_imgs[int(from_slice): int(to_slice)]
                neg_img = neg_imgs[int(from_slice): int(to_slice)]
                
                all_img = torch.cat((anc_img, pos_img, neg_img)) # Must be a tuple of Torch Tensors
                
                if model_architecture == "pointnetTriplet": 
                    anc_embeddings, pos_embeddings, neg_embeddings, m3x3, m64x64, model = forward_pass_Pointnet(
                        data=all_img,
                        model=model,
                        batch_size=batch_size,
                        accum_iter = batch_accum,
                        device = device
                        )
                else :
                    anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                        data=all_img,
                        model=model,
                        batch_size=batch_size,
                        accum_iter = batch_accum,
                        device = device
                        )      
                    
                pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
                neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

                if use_semihard_negatives:
                    # Semi-Hard Negative triplet selection
                    #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                    #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
                    first_condition = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                    second_condition = (pos_dists < neg_dists).cpu().numpy().flatten()
                    all = (np.logical_and(first_condition, second_condition))
                    valid_triplets = np.where(all == 1)
                else:
                    # Hard Negative triplet selection
                    #  (negative_distance - positive_distance < margin)
                    #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
                    all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                    valid_triplets = np.where(all == 1)

                anc_valid_embeddings = anc_embeddings[valid_triplets]
                pos_valid_embeddings = pos_embeddings[valid_triplets]
                neg_valid_embeddings = neg_embeddings[valid_triplets]

                # Calculate triplet loss
                
                
                if model_architecture == "pointnetTriplett":
                    triplet_loss = TripletLossPointnet(batch_size*3, alpha = 0.0001, margin=margin).forward(
                        anchor=anc_valid_embeddings,
                        positive=pos_valid_embeddings,
                        negative=neg_valid_embeddings,
                        m3x3 = m3x3, 
                        m64x64 = m64x64
                        )
                else:
                    triplet_loss = TripletLoss(margin=margin).forward(
                        anchor=anc_valid_embeddings,
                        positive=pos_valid_embeddings,
                        negative=neg_valid_embeddings
                        )
                # Calculating number of triplets that met the triplet selection method during the epoch
                num_valid_training_triplets += len(anc_valid_embeddings)    
                # Backward pass
                triplet_loss= triplet_loss / batch_accum
                if not torch.isnan(triplet_loss):
                    running_loss += triplet_loss.item()
                # triplet_loss = triplet_loss
                triplet_loss.backward()
                # print(list(model.parameters())[-1])
                
                if (i == (batch_accum-1)): 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer_model.step()
                    if not np.isnan(running_loss):
                        printf("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, running_loss/(batch_idx+1)))
                    else:
                        printf("Epoch: {}/{} - Loss: 0".format(epoch+1, epochs))
                    optimizer_model.zero_grad()

        # Print training statistics for epoch and add to log
        time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
        epoch_loss = running_loss / len(train_dataloader)
        accuracy_triplet = 100 - (num_triplets - num_valid_training_triplets)/num_triplets *100
        printf('Epoch {}:\tNumber of valid training triplets in epoch: {} , {:.2f}'.format(epoch,num_valid_training_triplets,accuracy_triplet))
        printf("Epoch: {}/{} - Loss: {:.4f} - Time:{}".format(epoch+1, epochs, epoch_loss,time_cost))
        
        # with open('logs/{}_log_triplet_batch{}.txt'.format(model_architecture,batch_size), 'a') as f:
        #     val_list = [
        #         epoch_loss,
        #         epoch,
        #         num_valid_training_triplets
        #     ]
        #     log = '\t'.join(str(value) for value in val_list)
        #     f.writelines(log + '\n')

        # Evaluation pass on LFW dataset
        best_distances = validate_Bosphorus(
            model=model,
            validate_dataloader=Bosphorus_dataloader,
            model_architecture=model_architecture,
            epoch=epoch,
            device=device,
            screen_logger=screen_logger
        )
        
        logger.append([epoch, optimizer_model.param_groups[0]['lr'],
                       epoch_loss, num_valid_training_triplets, np.mean(best_distances)])
        # Save model checkpoint
        state = {
            'epoch': epoch,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'best_distance_threshold': np.mean(best_distances)
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()
        # Save model checkpoint
        torch.save(state, '{}/{}_{}_{}.pt'.format(
                    args.checkpoint,
                    model_architecture,
                    message,
                    epoch,
                    
                )
            )
        
        scheduler.step()
    logger.close()


if __name__ == '__main__':
    main()
