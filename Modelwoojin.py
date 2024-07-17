import torch
import torch.nn as nn 
import torch.nn.functional as F

from zmq import NULL 
import numpy as np
np.set_printoptions(linewidth=np.inf)
torch.set_printoptions(profile="full")



#####################
### ECOC decoder  ###
#####################
class ecoc_decoder(nn.Module): 
    def __init__(self, W, flag=False, no_tanh=False): 
        super(ecoc_decoder, self).__init__()
        self.W = W
        self.no_tanh = no_tanh
        self.flag = flag
        self.activation1 = torch.nn.Tanh()   # Tanh to scale each bit in range [-1, 1]

    def forward(self, code):
        W_ = self.W.unsqueeze(0) 
        W_ = W_.repeat(W_.shape[0], 1, 1)
        W_ = torch.transpose(W_, dim0=1, dim1=2)
        if not self.no_tanh:
            code = self.activation1(code).unsqueeze(1)
        else: 
            code = code.unsqueeze(1)

        if (self.flag==True):
            print("direct output")
            out = code
        else:
            # print("matmul output")
            out = torch.matmul(code, W_.float()).squeeze(1)        
        return out 

class GaussianNoise(nn.Module):
    def __init__(self, stddev, input_shape):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev
        self.input_shape = input_shape
        self.noise = torch.zeros(*input_shape)

    def forward(self, x):
        self.noise.normal_(mean=0, std=self.stddev)
        # print("noise:", self.noise)
        # print(x.shape)
        return x + self.noise.to(x.device)


###############################
### ECOC Models for CIFAR10 ###
###############################
class ecoc_ensemble_no_bn(nn.Module):  
    def __init__(self, W, num_chunks, num_filter_ens, num_filter_ens_2, num_codes, dataset, activation='tanh'): 
        super(ecoc_ensemble_no_bn, self).__init__()
        self.W = W 
        self.num_chunks = num_chunks
        self.num_codes = num_codes
        self.activation = activation
        self.in_size = 0 # input size
        self.out = 0   # input to fc layer
        self.TRAIN_FLAG=1
       
        if dataset == "CIFAR10": 
            self.in_size = 3 
            self.out = 4          #filter 2 크기랑 맞춰서 바꾸자
        elif dataset == "MNIST": 
            self.in_size = 1
            self.out = 4
        elif dataset == "Fashion-MNIST": 
            self.in_size = 1
            self.out = 4
        
        # 4 chunks divided ensemble model
        self.models_ensemble = nn.ModuleList(ecoc_no_bn(W, num_chunks, num_filter_ens, num_filter_ens_2, num_codes, self.in_size, self.out, activation, dataset) for _ in range(num_chunks))

    def forward(self,x): 

        code = torch.cat([model(x, self.TRAIN_FLAG) for model in self.models_ensemble], dim=1)

            
        return code

class ecoc_no_bn(nn.Module): 
    def __init__(self, W, num_chunks, num_filter_ens, num_filter_ens_2, num_codes, in_size, out, activation, dataset): 
        super(ecoc_no_bn, self).__init__()
        self.W = W
        self.activation = activation                # activation function in our case is "tanh"
        self.num_codes = num_codes                  # number of bits in codes. Equivalent to the number of binary classifiers in the ecoc ensemble 
        self.num_chunks = num_chunks                # number of independent models in the ecoc ensemble
        self.num_filter_ens = num_filter_ens        # number of filters in the shared first conv layers 
        self.num_filter_ens_2 = num_filter_ens_2    # number of filters in the independant conv layers at the end of the model
        self.dataset_name = dataset
        
        if self.dataset_name == "CIFAR10": 
            self.input_shape = (3, 32, 32)
            self.gaussian_shape = (100, 1, 32, 32)
        elif self.dataset_name == "Fashion-MNIST": 
            self.input_shape = (1, 28, 28)
            self.gaussian_shape = (1, 28, 28)
        elif self.dataset_name == "MNIST": 
            self.input_shape = (100, 1, 28, 28)
            self.gaussian_shape = (100, 1, 28, 28)
        
        self.stddev = 0.032   #mnist 0.3 cifar 0.032
        self.blend_factor = 0.032
        
        self.gaussianNoise_x = GaussianNoise(self.stddev, self.input_shape)
        
        #self.grayscaler = Grayscaler.Grayscaler()
        #self.classBlender = ClassBlender.ClassBlender(self.blend_factor, 100)

        
        self.conv2D = torch.nn.Conv2d(in_size, num_filter_ens[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter_ens[0])
        self.conv2D_1 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter_ens[0])
        self.conv2D_2 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filter_ens[0])

        self.conv2D_3 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filter_ens[1])
        self.conv2D_4 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_filter_ens[1])
        self.conv2D_5 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(num_filter_ens[1])

        self.conv2D_6 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_filter_ens[2])
        self.conv2D_7 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(num_filter_ens[2])
        self.conv2D_8 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(num_filter_ens[2])

        self.independent_heads = nn.ModuleList(bloc_out_no_bn(num_filter_ens, num_filter_ens_2, out, num_codes) for _ in range(int(self.num_codes/self.num_chunks)))
        
    def forward(self, x, TRAIN_FLAG): 
        
        if(TRAIN_FLAG==1):
            x = self.gaussianNoise_x(x)
#         else:
#             print("inference")
        x = torch.clamp(x, -0.5, 0.5)
        
        x = F.relu(self.conv2D(x))
        x = self.bn1(x)     #cifar
        x = F.relu(self.conv2D_1(x))   #cifar
        x = self.bn2(x)
        x = F.relu(self.conv2D_2(x))
        x = self.bn3(x)

        x = F.relu(self.conv2D_3(x))
        x = self.bn4(x)     #cifar
        x = F.relu(self.conv2D_4(x))#cifar
        x = self.bn5(x)
        x = F.relu(self.conv2D_5(x))
        x = self.bn6(x)

        x = F.relu(self.conv2D_6(x)) 
        x = self.bn7(x)#cifar
        x = F.relu(self.conv2D_7(x))#cifar
        x = self.bn8(x)
        x = F.relu(self.conv2D_8(x))
        ##x = self.bn9(x)

        code = torch.cat([out_head(x) for out_head in self.independent_heads], dim=1)   
        return code

class bloc_out_no_bn(nn.Module): 
    def __init__(self,num_filter_ens, num_filter_ens_2, out, num_codes): 
        super(bloc_out_no_bn, self).__init__()
        self.conv2D_12 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)
        self.conv2D_13 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)    
        self.conv2D_14 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1) ## C'est le workaround que j'ai trouvé pour que la size en sortie de la couche soit de []
        
#         if dataset_name == "CIFAR10":
#             in_size = 3
#         elif dataset_name == "Fashion-MNIST" or "MNIST":
#             in_size = 1
            
        self.conv2D_15 = torch.nn.Conv2d(1, num_filter_ens_2[0], kernel_size=2, stride=2, padding=2)
        self.conv2D_16 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=2, padding=1)    
        self.conv2D_17 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=2, padding=1)
        
        self.conv2D_18 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)    
        self.conv2D_19 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)
        
        self.dense_1 = torch.nn.Linear(784, 16)  #196 mnist
        self.dense_2 = torch.nn.Linear(16, 8)
        self.dense_3 = torch.nn.Linear(8, 1) 
        
        self.dense_final = torch.nn.Linear(784,1)

    def forward(self, x): 
        x = F.relu(self.conv2D_12(x))                                 
        x = F.relu(self.conv2D_13(x))
        x_ = F.relu(self.conv2D_19(x))
        x_ = x_.flatten(1)
        x_ = F.relu(self.dense_1(x_))
        x_ = F.relu(self.dense_2(x_))
        output =F.tanh(self.dense_3(x_))
     
        return output

############################################
#### SIMPLE baseline vanilla network net ###
############################################

class simple(nn.Module): 
    def __init__(self, num_filter_ens, num_filter_ens_2, dataset): 
        super(simple, self).__init__()
        self.num_filter_ens = num_filter_ens        # number of filters in the shared first conv layers 
        self.num_filter_ens_2 = num_filter_ens_2    # number of filters in the independant conv layers at the end of the model
        if dataset == "CIFAR10": 
            in_size = 3 
            out = 256
        elif dataset == "Fashion-MNIST": 
            in_size = 1
            out = 64
        self.conv2D = torch.nn.Conv2d(in_size, num_filter_ens[0], kernel_size=5, stride=1, padding=2)
        self.conv2D_1 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=5, stride=1, padding=2)
        self.conv2D_2 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=3, stride=2, padding=1)
        self.conv2D_3 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.conv2D_4 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.conv2D_5 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=2, padding=1)
        self.conv2D_6 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.conv2D_7 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.conv2D_8 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=2, padding=1)
        self.conv2D_12 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)    
        self.conv2D_13 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=0) ## C'est le workaround que j'ai trouvé pour que la size en sortie de la couche soit de []
        self.dense_1 = torch.nn.Linear(out, 10) 

    def forward(self, x): 
        x = F.relu(self.conv2D(x))
        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = F.relu(self.conv2D_3(x))
        x = F.relu(self.conv2D_4(x))
        x = F.relu(self.conv2D_5(x))
        x = F.relu(self.conv2D_6(x)) 
        x = F.relu(self.conv2D_7(x))
        x = F.relu(self.conv2D_8(x))
        output = F.relu(self.conv2D_12(x))
        output = F.relu(self.conv2D_13(output))
        output = output.flatten(1)
        output = self.dense_1(output)
        
        return output 

class simple_ensemble(nn.Module): 
    def __init__(self, model_list, no_sotfmax=False): 
        super(simple_ensemble, self).__init__()
        self.models = model_list
        self.no_softmax = no_sotfmax
        self.activation = torch.nn.Softmax(dim=2) 
      
    def forward(self, x): 
        out = torch.stack([model(x) for model in self.models], dim=1)
        if not self.no_softmax:  
            out = self.activation(out)
        out = torch.sum(out, dim=1)    
        return out




#####################################
### ECOC with independent ResNets ###
#####################################
class ResNetECOC(nn.Module): 
    def __init__(self, W, nb_resnets, num_codes,dataset,activation = "tanh" ): 
        super(ResNetECOC, self).__init__()
        self.W = W 
        self.num_codes = num_codes
        self.num_chunks = nb_resnets
        self.activation = activation
        
        if dataset == "CIFAR10": 
            self.input_shape = (3, 32, 32)
            self.gaussian_shape = (1, 32, 32)
        elif dataset == "Fashion-MNIST" or dataset == "MNIST": 
            self.input_shape = (1, 28, 28)
            self.gaussian_shape = (1, 28, 28)
 
        self.stddev = 0.032
        self.gaussianNoise_x = GaussianNoise(self.stddev, self.input_shape)
        
        self.subnets = nn.ModuleList(ResNet(ResidualBlock, [3, 3, 3],dataset,num_classes=1) for _ in range(nb_resnets))
        if dataset == "CIFAR10": 
            self.in_channels = 3 
            out_channels = 64
        elif dataset == "Fashion-MNIST": 
            self.in_channels = 1
            out_channels = 64
        elif dataset == "STL-10": 
            self.in_channels = 3
            out_channels = 64         #몇으로 해야함?
        elif dataset == "MNIST": 
            self.in_channels = 1
            out_channels = 64       #몇으로 해야하지?
        elif dataset == "SVHN": 
            self.in_channels = 3
            out_channels = 64        #아웃풋채널은 뭘로 고르징

    def forward(self, x): 
        x = self.gaussianNoise_x(x)
        code = torch.cat([resnet(x) for resnet in self.subnets], dim=1)
        
        return code


############################
### Single ResNet models ###
############################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None): 
        super(ResidualBlock, self).__init__()    
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.downsample = downsample
    
    def forward(self, x): 
        residual = x 
        out = self.conv_1(x)
        out = self.bn1(out) 
        out = F.relu(out) 
        out = self.conv_2(out) 
        out = self.bn2(out)
        out = F.relu(out) 

        if self.downsample: 
            residual = self.downsample(x) 
        output = out + residual # new variable output prevents inplace operation with out.
        output = F.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, dataset_name,num_classes=1):
        """ 
        block: building block of network (here it is ResidualBlock)
        layers: [3,3,3] pour ResNet20
        num_classes: number of classes for output 
        """
        super(ResNet, self).__init__()
        self.in_channels = 16  
        if(dataset_name=="Fashion-MNIST"or dataset_name=="MNIST"):
          self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)   
        else:  
          self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)   
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        if(dataset_name=="Fashion-MNIST"or dataset_name=="MNIST"):
          self.avg_pool = nn.AvgPool2d(7)
        else:
          self.avg_pool = nn.AvgPool2d(8)
        if(dataset_name=="STL-10"):
          self.fc =nn.Linear(576,num_classes)
        else:
          self.fc = nn.Linear(64, num_classes)
      

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out