
import torch
import torch.nn as nn # this is needed for the pretrained model to load correctly, even though it is not used directly in this file
import numpy as np
from IPython import embed

from .base_color import *

class ECCVGenerator(BaseColor): #BaseColor is used for the normalization and unnormalization functions, as well as the centering and normalization constants
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__() 

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),] 
        model1+=[nn.ReLU(True),] # Relu is used in the pretrained model, so we need to use it here as well for the pretrained weights to load correctly
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained): # the pretrained weights are from the original authors of the ECCV 2016 paper, and are available on their website. We load them using torch.hub, which allows us to load the weights directly from the URL. The weights are in a .pth file, which is a common format for PyTorch model weights. The map_location='cpu' argument ensures that the weights are loaded onto the CPU, which is important for compatibility with different hardware setups. The check_hash=True argument verifies the integrity of the downloaded file by checking its hash against the expected value, ensuring that the file has not been corrupted during download.
		import torch.hub as hub
		model.load_state_dict(hub.load_state_dict_from_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model


# what is cnn ?
# answer : CNN stands for Convolutional Neural Network. It is a type of deep learning model that is particularly effective for processing data that has a grid-like topology, such as images. CNNs use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data, making them well-suited for tasks like image classification, object detection, and image colorization.

# what are all these models and layers in the code above ?
# answer : The code above defines a neural network architecture for image colorization based on the ECCV 2016 paper. The architecture consists of several convolutional layers, ReLU activation functions, batch normalization layers, and a final output layer that produces the colorized image. The model is structured in a way that allows it to learn complex features from the input grayscale image and generate the corresponding color information. Each "model" variable (model1, model2, etc.) represents a different stage of the network, with increasing depth and complexity as the data flows through the layers. The final output is obtained by applying a softmax function to the last convolutional layer's output, followed by a convolutional layer that maps the 313 color classes to the 2 color channels (ab) in the Lab color space.

# how are all models combined here ?
# answer : The models are combined sequentially in the forward method of the ECCVGenerator class. The input grayscale image (input_l) is passed through each model in order, with the output of one model serving as the input to the next. The final output from the last model (model8) is then processed through a softmax function and a final convolutional layer to produce the colorized output. The upsample4 layer is used to resize the output to match the original input size, as the convolutional layers reduce the spatial dimensions of the data. The combination of these models allows the network to learn and extract features at multiple levels of abstraction, ultimately enabling it to generate accurate colorizations from the input grayscale image.
