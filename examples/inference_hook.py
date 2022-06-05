import matplotlib.pyplot as plt
import numpy as np
import torch

from cpu import HookBase


class InferenceHook(HookBase):
    """Visualize the inference results of some randomly chosen images after each epoch."""

    def __init__(self, test_dataset, num=6):
        self.test_dataset = test_dataset
        self.num = num
        assert self.num % 2 == 0

    def after_epoch(self):
        model = self.trainer.model
        model.eval()
        ids = np.random.permutation(len(self.test_dataset))[:self.num]
        for i in range(self.num):
            id = ids[i]
            # img: [1, 28, 28], target: int
            img, target = self.test_dataset[id]
            # img: [1, 1, 28, 28]
            img = img.unsqueeze(0)
            # target: [1]
            target = torch.tensor([target])
            output = model((img, target))
            pred = output.argmax(dim=1).squeeze()

            plt.subplot(2, self.num // 2, i + 1)
            plt.tight_layout()
            plt.imshow(img[0][0], cmap='gray')
            plt.title(f"pred: {pred}, gt: {target[0]}")
            plt.xticks([])
            plt.yticks([])
        plt.show()
