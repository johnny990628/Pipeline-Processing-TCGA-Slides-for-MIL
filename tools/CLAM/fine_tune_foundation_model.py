# 導入必要的庫
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import h5py
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
from models.conch import create_model_from_pretrained
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import openslide

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SurvLabelTransformer(object):
    """
    SurvLabelTransformer: create label of survival data for model training.
    """
    def __init__(self, path_label, column_t='t', column_e='e', verbose=True):
        super(SurvLabelTransformer, self).__init__()
        self.path_label = path_label
        self.column_t = column_t
        self.column_e = column_e
        self.column_label = None
        self.full_data = pd.read_csv(path_label, dtype={'patient_id': str, 'pathology_id': str})
        
        self.pat_data = self.to_patient_data(self.full_data, at_column='patient_id')
        self.min_t = self.pat_data[column_t].min()
        self.max_t = self.pat_data[column_t].max()
        if verbose:
            print('[surv label] at patient level')
            print('\tmin/avg/median/max time = {}/{:.2f}/{}/{}'.format(self.min_t, 
                self.pat_data[column_t].mean(), self.pat_data[column_t].median(), self.max_t))
            print('\tratio of event = {}'.format(self.pat_data[column_e].sum() / len(self.pat_data)))

    def to_patient_data(self, df, at_column='patient_id'):
        df_gps = df.groupby('patient_id').groups
        df_idx = [i[0] for i in df_gps.values()]
        return df.loc[df_idx, :]

    def to_continuous(self, column_label='y'):
        print('[surv label] to continuous')
        self.column_label = [column_label]

        label = []
        for i in self.pat_data.index:
            if self.pat_data.loc[i, self.column_e] == 0:
                label.append(-1 * self.pat_data.loc[i, self.column_t])
            else:
                label.append(self.pat_data.loc[i, self.column_t])
        self.pat_data.loc[:, column_label] = label
        
        return self.pat_data

    def to_discrete(self, bins=4, column_label_t='y_t', column_label_c='y_c'):
        """
        based on the quartiles of survival time values (in months) of uncensored patients.
        see Chen et al. Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images
        """
        print('[surv label] to discrete, bins = {}'.format(bins))
        self.column_label = [column_label_t, column_label_c]

        # c = 1 -> censored/no event, c = 0 -> uncensored/event
        self.pat_data.loc[:, column_label_c] = 1 - self.pat_data.loc[:, self.column_e]

        # discrete time labels
        df_events = self.pat_data[self.pat_data[self.column_e] == 1]
        _, qbins = pd.qcut(df_events[self.column_t], q=bins, retbins=True, labels=False)
        qbins[0] = self.min_t - 1e-5
        qbins[-1] = self.max_t + 1e-5

        discrete_labels, qbins = pd.cut(self.pat_data[self.column_t], bins=qbins, retbins=True, labels=False, right=False, include_lowest=True)
        self.pat_data.loc[:, column_label_t] = discrete_labels.values.astype(int)

        return self.pat_data

    def collect_slide_info(self, pids, column_label=None):
        if column_label is None:
            column_label = self.column_label

        sel_pids, pid2sids, pid2label = list(), dict(), dict()
        for pid in pids:
            sel_idxs = self.full_data[self.full_data['patient_id'] == pid].index
            if len(sel_idxs) > 0:
                sel_pids.append(pid)
                pid2sids[pid] = list(self.full_data.loc[sel_idxs, 'pathology_id'])
                
                pat_idx = self.pat_data[self.pat_data['patient_id'] == pid].index[0]
                pid2label[pid] = list(self.pat_data.loc[pat_idx, column_label])

            else:
                print('[warning] patient {} not found!'.format(pid))

        return sel_pids, pid2sids, pid2label

# 定義WSI數據集
class WSIDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, h5_dir, slide_dir, slide_ext='.svs', custom_transforms=None):
        self.csv_path = csv_path
        self.h5_dir = h5_dir
        self.slide_dir = slide_dir
        self.slide_ext = slide_ext
        self.custom_transforms = custom_transforms
        self.slide_data = self.load_slide_data()

    def load_slide_data(self):
        """Load slide data and survival information from CSV using SurvLabelTransformer"""
        # 初始化 SurvLabelTransformer
        surv_label = SurvLabelTransformer(self.csv_path, verbose=True)
        
        # 轉換為連續的生存標籤
        patient_data = surv_label.to_continuous(column_label='y')
        
        # 收集全部病理切片的信息
        full_data = pd.read_csv(self.csv_path, dtype={'patient_id': str, 'pathology_id': str})
        
        # 準備返回的數據
        slide_data = []
        for _, row in full_data.iterrows():
            pathology_id = row['pathology_id']
            patient_id = row['patient_id']
            
            # 獲取對應病人的標籤
            pat_idx = patient_data[patient_data['patient_id'] == patient_id].index[0]
            label = patient_data.loc[pat_idx, 'y']
            
            slide_data.append((pathology_id, label))
            
        print(f"Loaded {len(slide_data)} slides with survival data")
        return slide_data

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id, label = self.slide_data[idx]
        h5_path = os.path.join(self.h5_dir, 'patches', f"{slide_id}.h5")
        slide_path = os.path.join(self.slide_dir, f"{slide_id}{self.slide_ext}")
        
        # 使用Whole_Slide_Bag_FP加載WSI
        wsi = openslide.open_slide(slide_path)
        wsi_dataset = Whole_Slide_Bag_FP(file_path=h5_path, wsi=wsi, custom_transforms=self.custom_transforms)

        # 確保數據格式正確
        if len(wsi_dataset) > 0:
            first_item = wsi_dataset[0]
            if not isinstance(first_item[0], torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(first_item[0])}")
            print("Sample tensor shape:", first_item[0].shape)
    
        
        return wsi_dataset, torch.tensor(label, dtype=torch.float32)



# 加載預訓練模型
def load_pretrained_model(ckpt_path, target_patch_size):
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", 
        checkpoint_path=ckpt_path,
        force_image_size=target_patch_size,
    )
    return model, preprocess

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class SurvivalModel(nn.Module):
    def __init__(self, dropout=True, dims=[512,256,128,64,32], **kwargs):
        super(SurvivalModel, self).__init__()
        fc = [nn.Linear(dims[0], dims[1]), 
              nn.ReLU(),
              nn.Dropout(0.25),
              nn.Linear(dims[1], dims[2]), 
              nn.ReLU(),
              nn.Dropout(0.25),
              nn.Linear(dims[2], dims[3]), 
        ]
        
        attention_net = Attn_Net_Gated(L=dims[3], D=dims[3], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.out_layer = nn.Sequential(
                    nn.Linear(dims[3], dims[4]),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(dims[4], 1)
        )

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def forward(self, h, attention_only=False):
        h=h.squeeze(0)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)  # A: 1 * N h: N * 512 => M: 1 * 512
        # M = torch.cat([M, embed_batch], axis=1)
        risk_score = self.out_layer(M)  
        # Y_hat = torch.topk(risk_score, 1, dim=1)[1]
        result = {
            'risk_score': risk_score,
            'attention_raw': A_raw,
            'M': M
        }
        return risk_score

# 定義損失函數 (Cox部分似然損失)
class CoxLoss(nn.Module):
    """A partial likelihood estimation (called Breslow estimation) function in Survival Analysis.

    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    Note that it only suppurts survival data with no ties (i.e., event occurrence at same time).
    
    Args:
        y (Tensor): The absolute value of y indicates the last observed time. The sign of y 
        represents the censor status. Negative value indicates a censored example.
        y_hat (Tensor): Predictions given by the survival prediction model.
    """
    def __init__(self):
        super(CoxLoss, self).__init__()
        print('[setup] loss: a popular PLE loss in coxph')

    def forward(self, y_hat, y, device):
        T = torch.abs(y)
        E = (y > 0).int()

        n_batch = len(T)
        R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
        for i in range(n_batch):
            for j in range(n_batch):
                R_matrix_train[i, j] = T[j] >= T[i]

        train_R = R_matrix_train.float().to(device)
        train_ystatus = E.float().to(device)

        theta = y_hat.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

        return loss_nn


def extract_features(model, wsi_dataset):
    """
    從單個WSI數據集中提取視覺特徵
    Args:
        model: CONCH model
        wsi_dataset: 單個WSI數據集
    Returns:
        視覺特徵張量
    """
    # 創建自定義的collate函數來處理numpy數組和張量的混合
    def custom_collate(batch):
        imgs = torch.stack([item[0] for item in batch])  # 使用stack替代cat
        coords = np.vstack([item[1] for item in batch])
        return imgs, coords

    loader = DataLoader(
        wsi_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True
    )
    
    features = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:

            # 處理形狀問題
            if len(imgs.shape) == 5:  # [batch, 1, channel, height, width]
                imgs = imgs.squeeze(1)  # 移除多餘的維度，變成 [batch, channel, height, width]
            
            # 確保輸入張量的形狀正確 [batch_size, channels, height, width]
            if len(imgs.shape) != 4:
                raise ValueError(f"Expected 4D input tensor, got shape {imgs.shape}")
            
            imgs = imgs.to(device)
            # 使用模型的module屬性來獲取原始模型（如果使用了DataParallel）
            actual_model = model.module if isinstance(model, torch.nn.parallel.DataParallel) else model

             # 檢查並打印輸入形狀
            print("Input shape:", imgs.shape)

            model = model.to(device)

            # 使用視覺編碼器提取特徵
            output = model(imgs)  # 使用visual而不是forward_no_head
            
            # 如果輸出是tuple，取第一個元素（通常是特徵）
            if isinstance(output, tuple):
                vis_features = output[0]
            else:
                vis_features = output
                
            features.append(vis_features.cpu())
    
    return torch.cat(features, dim=0)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, accumulation_steps=169, save_dir='checkpoints'):

     # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,  # LoRA rank
        lora_alpha=16,
        target_modules=["qkv", "proj"],  # 需要根據你的模型架構調整
        lora_dropout=0.1,
    )
    
    # 將模型轉換為 LoRA 模型
    model = get_peft_model(model, lora_config)
    print_network(model) 

    feature_dim = 512
    survival_model = SurvivalModel(dims=[feature_dim, 256, 128, 64, 32]).to(device)

    criterion = CoxLoss().to(device)
    best_val_loss = float('inf')
    optimizer = optim.Adam(list(model.parameters()) + list(survival_model.parameters()), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        survival_model.train()
        
        total_loss = 0.0
        all_risk_scores = []
        all_labels = []
        
        # 直接遍歷數據集
        for i in range(len(train_loader.dataset)):
            wsi_dataset, label = train_loader.dataset[i]
            
            # Extract features
            features = extract_features(model, wsi_dataset)
            features = features.to(device)
            
            # Calculate risk scores
            risk_score = survival_model(features)
            
            all_risk_scores.append(risk_score)
            all_labels.append(label)
            
            # 每accumulation_steps個樣本更新一次
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader.dataset):
                # Stack collected tensors
                y = torch.stack(all_labels).to(device)
                y_hat = torch.cat(all_risk_scores)
                
                # Calculate loss
                loss = criterion(y_hat, y, device) / accumulation_steps
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader.dataset)}], Loss: {loss.item():.4f}")
                
                # Reset collectors
                all_risk_scores = []
                all_labels = []

        # 驗證階段
        model.eval()
        survival_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_loader.dataset)):
                wsi_dataset, label = val_loader.dataset[i]
                features = extract_features(model, wsi_dataset)
                features = features.to(device)
                label = label.to(device)
                
                risk_score = survival_model(features)
                loss = criterion(risk_score, label, device)
                val_loss += loss.item()
        
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 保存 LoRA 模型
            model.save_pretrained(os.path.join(save_dir, f'best_lora_model'))
            
            # 保存 survival model
            torch.save({
                'epoch': epoch,
                'model_state_dict': survival_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'best_survival_model.pth'))
            
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # 定期保存檢查點
        if (epoch + 1) % 5 == 0:  # 每5個epoch保存一次
            model.save_pretrained(os.path.join(save_dir, f'lora_checkpoint_epoch_{epoch+1}'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': survival_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'survival_checkpoint_epoch_{epoch+1}.pth'))
    
    return model, survival_model
    
# 設置參數
csv_path = "/data1/johnny99457/DSCA/data_split/tcga_luad_merged/tcga_luad_merged_path_full.csv"
h5_dir = "/data1/johnny99457/PATCHES/LUAD/tiles-5x-s448"
slide_dir = "/data1/johnny99457/DATASETS/TCGA/LUAD"
ckpt_path = "/data1/johnny99457/CLAM/checkpoints/conch/pytorch_model.bin"
target_patch_size = 224
batch_size = 1  # 每個batch只包含一個WSI
num_epochs = 10
learning_rate = 1e-4

# 加載預訓練模型
model, preprocess = load_pretrained_model(ckpt_path, target_patch_size)
# model = model.to(device)

# 創建數據集和數據載入器
dataset = WSIDataset(csv_path, h5_dir, slide_dir, custom_transforms=preprocess)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 使用最簡單的DataLoader設置
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True)


train_model(model, train_loader, val_loader, num_epochs, learning_rate)