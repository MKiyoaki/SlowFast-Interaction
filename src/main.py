import os.path

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import slowfast.models as models
from pytorchvideo.data import make_clip_sampler
from torchvision.transforms import Compose

from helper.configurations import video_dir, label_dir, data_dir
from helper.data_process import dataset_split
from interaction_classification_module import InteractionClassificationModule
from interaction_data_module import InteractionDataModule


def train(model, optimizer, criterion, train_loader, device):
    model.train()  # 将模型设置为训练模式
    train_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 创建一个进度条显示训练进度
    with tqdm(total=len(train_loader), desc="Training") as progress_bar:
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["video"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'loss': train_loss / (batch_idx + 1), 'accuracy': 100. * correct / total})
            progress_bar.update(1)

    # 计算平均训练损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    print(f"\nAverage Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def main():
    # Split dataset into train, validation, and test sets
    train_file, val_file, test_file = dataset_split(video_dir, label_dir, data_dir)

    # Initialize the model
    model = models.ResNet
    criterion = F.cross_entropy

    classy_module = InteractionClassificationModule(model=model)
    data_module = InteractionDataModule(
        data_dir=data_dir,
        batch_size=32,
        clip_duration=2,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    return 0


if __name__ == '__main__':
    main()
