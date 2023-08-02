# %%
import os
import h5py
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset
from models.pointmlp import pointMLPEliteTriplet, pointMLPTriplet
from losses.triplet_loss import TripletLoss
from torchsummary import summary
import datetime



os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_data_2():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_face_class = []
    
    h5_name =  os.path.join(DATA_DIR, 'point_cloud_original', 'face_data.h5')
    f = h5py.File(h5_name,'r')
    data = f["data"][:].astype('float32')
    label = f["pid"][:].astype('int64')
    face_class = f["class"][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_face_class.append(face_class)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_face_class = np.concatenate(all_face_class, axis=0)
    return all_data, all_label, all_face_class
    
def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def normalize_pointcloud(pointcloud):
    # Normalizing sampled point cloud.
    norm_point_cloud = pointcloud - np.mean(pointcloud, axis=0)
    norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
    return norm_point_cloud 
    
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

class TripletPoints_Dataset(Dataset):
    """Bosphorus original with mesh element sampling 10K points dataset."""
    def __init__(self, root_dir, training_dataset_csv_path, num_triplets, epoch, num_human_identities_per_batch=32,
                 triplet_batch_size=64, training_triplets_path=None, partition = "train"):
        """
        Args:

        root_dir: Absolute path to dataset.
        training_dataset_csv_path: Path to csv file containing the image paths inside the training dataset folder.
        num_triplets: Number of triplets required to be generated.
        epoch: Current epoch number (used for saving the generated triplet list for this epoch).
        num_generate_triplets_processes: Number of separate Python processes to be created for the triplet generation
                                          process. A value of 0 would generate a number of processes equal to the
                                          number of available CPU cores.
        num_human_identities_per_batch: Number of set human identities per batch size.
        triplet_batch_size: Required number of triplets in a batch.
        training_triplets_path: Path to a pre-generated triplet numpy file to skip the triplet generation process (Only
                                 will be used for one epoch).
        transform: Required image transformation (augmentation) settings.
        """

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #  VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #  forcing the 'name' column as being of type 'int' instead of type 'object')
        self.df = pd.read_csv(training_dataset_csv_path, dtype={'id': object, 'name': object, 'class': int})
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        self.epoch = epoch
        self.partition = partition
        
        # Modified here to bypass having to use pandas.dataframe.loc for retrieving the class name
        #  and using dataframe.iloc for creating the face_classes dictionary
        df_dict = self.df.to_dict()
        self.df_dict_class_name = df_dict["name"]
        self.df_dict_id = df_dict["id"]
        self.df_dict_class_reversed = {value: key for (key, value) in df_dict["class"].items()}

        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets()
        else:
            print("Loading pre-generated triplets file ...")
            self.training_triplets = np.load(training_triplets_path)

    def make_dictionary_for_face_class(self):
        """
            face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        """
        face_classes = dict()
        for idx, label in enumerate(self.df['class']):
            if label not in face_classes:
                face_classes[label] = []
            # Instead of utilizing the computationally intensive pandas.dataframe.iloc() operation
            face_classes[label].append(self.df_dict_id[idx])

        return face_classes

    def generate_triplets(self):
        triplets = []
        classes = self.df['class'].unique()
        face_classes = self.make_dictionary_for_face_class()

        print("\nGenerating {} triplets ...".format(self.num_triplets))
        num_training_iterations_per_process = self.num_triplets / self.triplet_batch_size
        progress_bar = tqdm(range(int(num_training_iterations_per_process)))  # tqdm progress bar does not iterate through float numbers

        for training_iteration in progress_bar:

            """
            For each batch: 
                - Randomly choose set amount of human identities (classes) for each batch
            
                  - For triplet in batch:
                      - Randomly choose anchor, positive and negative images for triplet loss
                      - Anchor and positive images in pos_class
                      - Negative image in neg_class
                      - At least, two images needed for anchor and positive images in pos_class
                      - Negative image should have different class as anchor and positive images by definition
            """
            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)

            for triplet in range(self.triplet_batch_size):

                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)

                while len(face_classes[pos_class]) < 2: #make sute the class have more than 1 data
                    pos_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:       #make sure the class positive and negative is not the same
                    neg_class = np.random.choice(classes_per_batch)
                #find last index and name of class
                pos_name_index = self.df_dict_class_reversed[pos_class]
                pos_name = self.df_dict_class_name[pos_name_index]

                neg_name_index = self.df_dict_class_reversed[neg_class]
                neg_name = self.df_dict_class_name[neg_name_index]

                if len(face_classes[pos_class]) == 2: #if the positivce class have 2 data
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                else:
                    ianc = np.random.randint(0, len(face_classes[pos_class]))
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

                    while ianc == ipos: #make sure anchor not same as positive
                        ipos = np.random.randint(0, len(face_classes[pos_class]))

                ineg = np.random.randint(0, len(face_classes[neg_class]))

                triplets.append(
                    [
                        face_classes[pos_class][ianc],
                        face_classes[pos_class][ipos],
                        face_classes[neg_class][ineg],
                        pos_class,
                        neg_class,
                        pos_name,
                        neg_name
                    ]
                )

        print("Saving training triplets list in 'data/generated_triplets' directory ...")
        file_output = str('data/generated_triplets/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                self.epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            ))
        # file_output = os.path.join(BASE_DIR,file_output)
        np.save(file_output,
            triplets
        )
        print("Training triplets' list Saved!\n")

        return triplets
        
    def __len__(self):
        return len(self.training_triplets)
    
    def __getitem__(self, idx):
        
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
        
        anc_pcd = os.path.join(self.root_dir, str(pos_name), str(anc_id)+ ".ply")
        pos_pcd = os.path.join(self.root_dir, str(pos_name), str(pos_id)+ ".ply")
        neg_pcd = os.path.join(self.root_dir, str(neg_name), str(neg_id)+ ".ply")
        
        pcd = o3d.io.read_point_cloud(anc_pcd)
        anc_pcd=  np.asarray(pcd.points)
        pcd = o3d.io.read_point_cloud(pos_pcd)
        pos_pcd =  np.asarray(pcd.points)
        pcd = o3d.io.read_point_cloud(neg_pcd)
        neg_pcd =  np.asarray(pcd.points)
        
        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
        
        #normalize pointcloud
        anc_pcd = normalize_pointcloud(anc_pcd)
        pos_pcd = normalize_pointcloud(pos_pcd)
        neg_pcd = normalize_pointcloud(neg_pcd)
        
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            #randomize translation
            anc_pcd = translate_pointcloud(anc_pcd)
            pos_pcd = translate_pointcloud(pos_pcd)
            neg_pcd = translate_pointcloud(neg_pcd)
            # randomize point index
            np.random.shuffle(anc_pcd)
            np.random.shuffle(pos_pcd)
            np.random.shuffle(neg_pcd)
            
          
        sample = {
            'anc_img': anc_pcd,
            'pos_img': pos_pcd,
            'neg_img': neg_pcd,
            'pos_class': pos_class,
            'neg_class': neg_class
        }
        
        return sample

def forward_pass(data, model, batch_size, device, accum_iter):
    data = data.to(device)
    data = data.permute(0, 2, 1)
    embeddings = model(data)
    

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:int(batch_size/accum_iter)]
    pos_embeddings = embeddings[int(batch_size/accum_iter): int(batch_size/accum_iter * 2)]
    neg_embeddings = embeddings[int(batch_size/accum_iter * 2):]

    return anc_embeddings, pos_embeddings, neg_embeddings, model

if __name__ == '__main__':
    from torch.nn.modules.distance import PairwiseDistance
    import torch.optim as optim
    l2_distance = PairwiseDistance(p=2)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataroot = os.path.join(BASE_DIR,"train_file") 
    csv_path = os.path.join(BASE_DIR,"BosphorusCustom.csv")
    iterations_per_epoch = 20                                                                        
    batch_size = 32
    num_human_identities_per_batch = 32
    batch_accum = 1
    _training_triplets_path = None
    device = "cuda"
    
    start_epoch =0
    epochs = 50
    margin = 0.5
    embedding_dimension =4096
    model_architecture = "pointMLPElite"
    # model_architecture = "pointMLP"
    model = pointMLPEliteTriplet().to(device)
    # model = pointMLPTriplet().to(device)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_triplets = iterations_per_epoch * batch_size
   
    for epoch in range(start_epoch , epochs):
        running_loss = []
        accuracy_triplet = 0
        time_cost = datetime.datetime.now()
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(p=2)
        _training_triplets_path = None

        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletPoints_Dataset(
                root_dir= dataroot,
                training_dataset_csv_path= csv_path,
                num_triplets=num_triplets,
                num_human_identities_per_batch=num_human_identities_per_batch,
                triplet_batch_size=batch_size,
                epoch=epoch,
                training_triplets_path=_training_triplets_path
            ),
            batch_size=batch_size,
            num_workers=8,
            shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
        )
        model.train()
        progress_bar = enumerate(tqdm(train_dataloader))
        
        for batch_idx, (batch_sample) in progress_bar:
            # Forward pass - compute embeddings
            anc_imgs = batch_sample['anc_img']
            pos_imgs = batch_sample['pos_img']
            neg_imgs = batch_sample['neg_img']
            
            for i in range (batch_accum):

                from_slice = i*(batch_size/batch_accum)
                to_slice = (i+1)*(batch_size/batch_accum)
                anc_img = anc_imgs[int(from_slice): int(to_slice)]
                pos_img = pos_imgs[int(from_slice): int(to_slice)]
                neg_img = neg_imgs[int(from_slice): int(to_slice)]
                all_img = torch.cat((anc_img, pos_img, neg_img))
                anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                data=all_img,
                model=model,
                batch_size=batch_size,
                device = device,
                accum_iter = batch_accum
            )   
                    
                pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
                neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)
    
                all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                valid_triplets = np.where(all == 1)
    
                anc_valid_embeddings = anc_embeddings[valid_triplets]
                pos_valid_embeddings = pos_embeddings[valid_triplets]
                neg_valid_embeddings = neg_embeddings[valid_triplets]
            
                # Calculate triplet loss
                triplet_loss = TripletLoss(margin).forward(
                anchor=anc_valid_embeddings,
                positive=pos_valid_embeddings,
                negative=neg_valid_embeddings
                )
            
                # Calculating number of triplets that met the triplet selection method during the epoch
                num_valid_training_triplets += len(anc_valid_embeddings)    
                # Backward pass
                triplet_loss = triplet_loss / batch_accum
                
                triplet_loss.backward()
                # print(list(model.parameters())[-1])
                
                if (i == (batch_accum-1)): 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    running_loss.append(triplet_loss.cpu().detach().numpy())
                    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
                    optimizer.zero_grad()
                    
                    
        # Print training statistics for epoch and add to log
        accuracy_triplet = (num_triplets - num_valid_training_triplets)/num_triplets *100
        print('Epoch {}:\tNumber of valid training triplets in epoch: {} , {:.4f}'.format(epoch,num_valid_training_triplets,accuracy_triplet))
        
        # Print training statistics for epoch and add to log
        with open('logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [ epoch, num_valid_training_triplets]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')
        
        # Save model checkpoint
        state = {
            'epoch': epoch,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer.state_dict()
            # 'best_distance_threshold': np.mean(best_distances)
        }

        torch.save(state, 'model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(
                model_architecture,
                epoch
            )
        )

        # Evaluation pass on LFW dataset
            
            
