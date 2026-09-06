import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DepthDataset(Dataset):

    def __init__(self, rgb_root, depth_root):

        self.samples = []

        for session in os.listdir(rgb_root):
            session_path = os.path.join(rgb_root, session)

            if not os.path.isdir(session_path):
                continue

            for sub in os.listdir(session_path):
                sub_path = os.path.join(session_path, sub)

                if os.path.isdir(sub_path):
                    for f in os.listdir(sub_path):

                        if f.endswith(".png"):
                            rgb = os.path.join(sub_path, f)
                            file_name = os.path.splitext(f)[0]
                            depth_file = f"{file_name}_depth.npy"
                            depth = os.path.join(depth_root, session, depth_file)

                            if os.path.exists(depth):
                                self.samples.append((rgb, depth))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        rgb_path, depth_path = self.samples[idx]

        img = cv2.imread(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (180,180))

        depth = np.load(depth_path)
        depth = cv2.resize(depth, (180,180))

        img = torch.tensor(img/255.).permute(2,0,1).float()
        depth = torch.tensor(depth).unsqueeze(0).float()

        return img, depth