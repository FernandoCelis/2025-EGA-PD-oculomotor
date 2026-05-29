# ---------------------- Configuration ----------------------
import os
import json
import logging
import random
import csv
import numpy as np
from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------- Define PARAMETERS ----------------------
PARAMETERS = {
    "GPU_DEVICE_NUMBER": "0",
    "data_root_dir": ".../EGA_Enhanced_Geodesic_Attention/data/monocular_frames",
    "output_root_dir": ".../EGA_Enhanced_Geodesic_Attention/outputs/",
    "attention_mode": "cross",
    "data_augmentation": True,
    "step_save_checkpoints": True,
    "subsampling_step": 6,
    "batch_size": 10,
    "epoch": 80,
    "transform_name": "original",
    'optimizer': {
        'kind': 'Adam',
        "lr": 0.001,
        'lr_others': 0.00001,
        "momentum": 0.9,
    },
    "seed": 2000,
    "EarlyStopping": False
}

_MODE_CONFIG = {
    "self":  {"name_model": "EGA",                       "name_dataset": "EyeVideoDataset"},
    "cross": {"name_model": "CrossATT_64Bire_32AttLEFT", "name_dataset": "LateralVideoDataset"},
}
PARAMETERS["name_model"]   = _MODE_CONFIG[PARAMETERS["attention_mode"]]["name_model"]
PARAMETERS["name_dataset"] = _MODE_CONFIG[PARAMETERS["attention_mode"]]["name_dataset"]

# ---------------------- Configure CUDA ----------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = PARAMETERS["GPU_DEVICE_NUMBER"]

# ---------------------- Import PyTorch & Libraries ----------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom Libraries
from libs.spdnetwork.optimizers import MixOptimizer
import data.dataloader as datasets
from data.dataloader import get_5kf_fold_partitions
from data.transforms import build_transforms
import models
from utils.utils import get_date, seed_worker, count_parameters

# ---------------------- Set Seed for Reproducibility ----------------------
torch.manual_seed(PARAMETERS["seed"])
random.seed(PARAMETERS["seed"])

# ---------------------- Define Transforms ----------------------
train_transforms, val_transforms = build_transforms(
    PARAMETERS["transform_name"],
    image_size=[64, 64]
)

# ---------------------- Create Output Folder ----------------------
lr_save_name = str(PARAMETERS['optimizer']['lr']).replace("0.", "p")
lro_save_name = str(PARAMETERS['optimizer']['lr_others']).replace("0.", "p")

PARAMETERS["output_name_model"] = f"{PARAMETERS['name_model']}_{PARAMETERS['name_dataset']}_bs-{PARAMETERS['batch_size']}_lr-{lr_save_name}_{get_date()}"
PARAMETERS["output_dir"] = os.path.join(PARAMETERS['output_root_dir'], "models", PARAMETERS["output_name_model"])

print(f"Creating folder: {PARAMETERS['output_dir']}")
os.makedirs(PARAMETERS['output_dir'], exist_ok=True)

# ---------------------- Configure Logging ----------------------
logging.basicConfig(
    filename=os.path.join(PARAMETERS['output_dir'], "output.log"),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)


# ---------------------- Select Device ----------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
    logging.info(f'Using GPU: {torch.cuda.get_device_name()}')
    logging.info(f'CUDA Visible devices: {os.getenv("CUDA_VISIBLE_DEVICES")}')
else:
    device = torch.device('cpu')
    logging.info('Failed to find GPU, using CPU instead.')

dtype = torch.double

#----------------------------------------------------------------
# Saving parameters
#----------------------------------------------------------------
json.dump(PARAMETERS, open(os.path.join(PARAMETERS["output_dir"],"parameters.json"), "w"))
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#                      5K - FOLD VALIDATION
#----------------------------------------------------------------------
#----------------------------------------------------------------------
    
    
# Create the CSV file and write the headers
with open(os.path.join(PARAMETERS['output_dir'],"trainingVal_info.csv"), 'w', newline='') as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(["Fold_index",'Training/Val','Epoch', 'Batch', 'Loss', 'names_patients', "labels", "predictions"])
    
    data_folds_dict = get_5kf_fold_partitions(seed = PARAMETERS["seed"])

    for idx_fold in range(1,6):
        logging.info(f"Starting fold {idx_fold}")
        train_names = data_folds_dict[f"fold_{idx_fold}_train"]
        val_names = data_folds_dict[f"fold_{idx_fold}_val"]

        logging.info(f"Train names: {data_folds_dict[f'fold_{idx_fold}_train']}")
        logging.info(f"Val names: {data_folds_dict[f'fold_{idx_fold}_val']}")

        train_data = getattr(datasets, PARAMETERS['name_dataset'])(
                                    root_dir = PARAMETERS["data_root_dir"], 
                                    names_patients = train_names,
                                    data_augmentation = PARAMETERS["data_augmentation"],
                                    subsampling_step =  PARAMETERS["subsampling_step"],
                                    transform = train_transforms)

        val_data = getattr(datasets, PARAMETERS['name_dataset'])(
                                    root_dir = PARAMETERS["data_root_dir"],
                                    names_patients = val_names,
                                    data_augmentation = False,
                                    subsampling_step =  PARAMETERS["subsampling_step"],
                                    transform = val_transforms)

        # Create the dataloaders and seeding the workers
        g = torch.Generator()
        g.manual_seed(PARAMETERS["seed"])

        train_loader = DataLoader(dataset = train_data, 
                                batch_size = PARAMETERS["batch_size"], 
                                worker_init_fn = seed_worker,
                                shuffle = True)
        val_loader = DataLoader(dataset = val_data, 
                                batch_size = PARAMETERS["batch_size"], 
                                worker_init_fn = seed_worker,
                                shuffle = False)    
        
        logging.info(f"Train loader size: {len(train_loader)} with batch_size: {PARAMETERS['batch_size']}")
        logging.info(f"Val loader size: {len(val_loader)} with batch_size: {PARAMETERS['batch_size']}")
        #Define the model 
        model = getattr(models, PARAMETERS['name_model'])(dtype=dtype, device=device)

        num_params = count_parameters(model)
        logging.info(f"Model: {PARAMETERS['name_model']} - Trainable Parameters: {num_params:,}")
        
        criterion = nn.BCEWithLogitsLoss()

        #----------------------------------------------------------------
        # OPTIMIZERS 
        #----------------------------------------------------------------
        optimizer_mapping = {
            'SGD': (torch.optim.SGD, {
                                'lr': PARAMETERS['optimizer']['lr'],
                                'lr': PARAMETERS['optimizer']['lr_others']
                                }),
            'Adam': (torch.optim.Adam, {
                                'lr': PARAMETERS['optimizer']['lr'],
                                'lr': PARAMETERS['optimizer']['lr_others']
                                }),
            'RMSprop': (torch.optim.RMSprop, {
                                'lr': PARAMETERS['optimizer']['lr'], 
                                'lr': PARAMETERS['optimizer']['lr_others'],                           
                                'momentum': PARAMETERS['optimizer']['momentum']
            })
        }
        optimizer_class, optimizer_args = optimizer_mapping[PARAMETERS['optimizer']['kind']]

        optimizer = MixOptimizer(model.parameters(), optimizer=optimizer_class, **optimizer_args)

        model.to(device)
        model.type(dtype)

        for epoch in range(PARAMETERS["epoch"]):
            train_loss = 0.0
            logging.info(f"Starting Trainig Epoch {epoch}")
            for i, (inputs, labels, name_patients) in enumerate(train_loader):

                labels = labels.to(device)
                if isinstance(inputs, (list, tuple)):
                    inputs = tuple(t.to(device).type(dtype).permute(0, 2, 1, 3, 4) for t in inputs)
                    batch_size = inputs[0].size(0)
                else:
                    inputs = inputs.to(device).type(dtype).permute(0, 2, 1, 3, 4)
                    batch_size = inputs.size(0)

                optimizer.zero_grad()

                outputs = model(inputs)
                outputs = torch.squeeze(outputs, 1)
                loss = criterion(outputs, labels.float())

                labels = labels.detach().cpu().numpy()
                output_prob = torch.sigmoid(outputs).detach().cpu().numpy()
                name_patients = list(name_patients)

                preds = np.round(output_prob)
                acc_score = accuracy_score(y_true=labels, y_pred=preds)

                logging.info(f"labels: {labels}, probs: {output_prob}, preds: {preds}")

                loss.backward()
                optimizer.step()

                avg_batch_loss = loss.item()
                train_loss += avg_batch_loss * batch_size

                writer_csv.writerow([idx_fold,'Training',epoch, i, avg_batch_loss, name_patients, labels, output_prob])
            logging.info(f"Epoch: {epoch} - Train loss: {train_loss/len(train_loader)}")

            logging.info(f"Starting Validation Epoch {epoch}")
            #Validation
            model.eval()
            
            val_loss = 0.0
            best_val_loss = float('inf')
            with torch.no_grad():
                for i, (inputs, labels, name_patients) in enumerate(val_loader):
                    labels = labels.to(device)
                    if isinstance(inputs, (list, tuple)):
                        inputs = tuple(t.to(device).type(dtype).permute(0, 2, 1, 3, 4) for t in inputs)
                        batch_size = inputs[0].size(0)
                    else:
                        inputs = inputs.to(device).type(dtype).permute(0, 2, 1, 3, 4)
                        batch_size = inputs.size(0)

                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs, 1)
                    loss = criterion(outputs, labels.float())
                    logging.info(f"Epoch [{epoch}/{PARAMETERS['epoch']}] - Step [{i}/{len(val_loader)-1}] - Val Loss: {loss.item():.4f}")

                    labels = labels.detach().cpu().numpy()
                    output_prob = torch.sigmoid(outputs).detach().cpu().numpy()
                    name_patients = list(name_patients)

                    preds = np.round(output_prob)
                    acc_score = accuracy_score(y_true=labels, y_pred=preds)

                    avg_batch_loss = loss.item()
                    val_loss += avg_batch_loss * batch_size
                    
                    writer_csv.writerow([idx_fold,'Val',epoch, i, avg_batch_loss, '', name_patients, labels, output_prob])
                    if loss.item() < best_val_loss:
                        best_val_loss = loss.item()
                        torch.save(model.state_dict(), os.path.join(PARAMETERS["output_dir"], f"fold-{idx_fold}_best.pt"))
                        logging.info(f"New best model saved for fold {idx_fold} at epoch {epoch}, step {i} with val_loss: {best_val_loss:.4f}")
                        logging.info(f"Epoch: {epoch} - Val loss: {val_loss/len(val_loader)}")
            
    
            torch.save(model.state_dict(), os.path.join(PARAMETERS["output_dir"], f"fold-{idx_fold}_last.pt"))
            logging.info(f"Saved last epoch model for fold {idx_fold}")


            logging.info(f"Epoch: {epoch} - Val loss: {val_loss/len(val_loader)}")
            
