from util.instructor_solution_guide import ResNetClassifier,CustomCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs,logger=None):
        super(Model, self).__init__()

        self.logger = logger
        if logger is not None:
            self.logger.info('Initializing ResNetClassifier')

        self.loss_type = 'l2'
        self.model = ResNetClassifier(num_classes=2)

    def forward(self, x):
        return self.model(x)


