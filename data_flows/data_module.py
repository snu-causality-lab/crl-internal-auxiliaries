import os
from typing import Optional
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm  

_REPO_ROOT = Path(__file__).resolve().parents[1]

class ImageDataModule(LightningDataModule):
    def __init__(self,dataset, batch_size, num_workers, log_dir, selected_idx= [1],observed_idx= [2]):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir= log_dir
        self.selected_idx = selected_idx  
        self.observed_idx = observed_idx  
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset = dataset
        self.data_prepared=False

    def load_data_from_folder(self, folder_name):
        images = []
        v_values = []


        transform = transforms.ToTensor()  


        if self.dataset == "flow":
            folder_path = _REPO_ROOT / "data_flows" / "causal_data" / "flow_noise" / folder_name
        elif self.dataset == "pendulum":
            folder_path = _REPO_ROOT / "data_pendulum" / "causal_data" / "pendulum" / folder_name
        else:
            raise ValueError(f"Unknown image dataset: {self.dataset}")
        if not folder_path.exists():
            raise FileNotFoundError(
                f"Missing image dataset directory: {folder_path}. "
                "Generate the image datasets before running image experiments."
            )
        for filename in tqdm(sorted(os.listdir(folder_path))):  
            if filename.endswith(".png"):

                file_path = folder_path / filename

   
                image = Image.open(file_path) #.convert("L")
                images.append(transform(image))

   
                parts = filename.split("_")
                v = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4].split(".png")[0])]
                v_values.append(torch.tensor(v, dtype=torch.float))


        
        x = torch.stack(images)  
        v = torch.stack(v_values)  
        v_mean = v.mean(dim=0, keepdim=True)  # Mean values per column
        v_std = v.std(dim=0, keepdim=True)  # Std deviation per column
        v = (v - v_mean) / v_std  # Standardization
        return x, v

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_prepared:
   
            train_x, train_v = self.load_data_from_folder("train")

 
            z_o_s_train = train_v[:, self.selected_idx]
            z_o_train = train_v[:, self.observed_idx]
            train_dataset = TensorDataset(train_x, train_v, z_o_s_train, z_o_train)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])


            test_x, test_v = self.load_data_from_folder("test")
            z_o_s_test = test_v[:, self.selected_idx]
            z_o_test = test_v[:, self.observed_idx]
            self.test_dataset = TensorDataset(test_x, test_v, z_o_s_test, z_o_test)

            self.data_prepared = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
