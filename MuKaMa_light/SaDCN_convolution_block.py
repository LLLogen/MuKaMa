# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch import nn
import warnings

warnings.filterwarnings("ignore")

"""
This code is mainly the deformation process of our SaDCNConv
"""


class SaDCNConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(SaDCNConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset使用卷积层 <offset_conv> 学习可变形卷积的偏移量
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)#输入通道数为 in_ch，输出通道数为 2 * kernel_size，卷积核大小为 3，padding 为 1
        self.bn = nn.BatchNorm2d(2 * kernel_size)# 批归一化层，对 2 * kernel_size 个通道进行批归一化
        self.kernel_size = kernel_size

        # two types of the SaDCNConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),#(15,1)
            stride=(kernel_size, 1),
            padding=0,
        )#沿着 x 轴进行的动态蛇卷积，输入通道数为 in_ch，输出通道数为 out_ch,
        #卷积核大小为 (kernel_size, 1)，步长为 (kernel_size, 1)，padding 为 0
        self.dsc_conv_y = nn.Conv2d(
            in_ch,#输入数据的通道数
            out_ch,#输出数据的通道数
            kernel_size=(1, kernel_size),#卷积核大小
            stride=(1, kernel_size),#步长大小
            padding=0,#填充的大小
        )# 沿着 y 轴进行的动态蛇卷积，输入通道数为 in_ch，输出通道数为 out_ch
        # 卷积核大小为 (1, kernel_size)，步长为 (1, kernel_size)，padding 为 0

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)# 组归一化层，将 out_ch 个通道分为 out_ch // 4 组进行归一化
        self.relu = nn.ReLU(inplace=True)# ReLU 激活函数

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

        '''================================================================================================'''

        self.Weight = nn.Linear(6, 2, bias=False)#线性层,6映射成2
        self.bias = nn.Parameter(torch.ones([2]))#定义了一个可学习的偏置项，并将其赋值给模型的 self.bias 属性，以便在模型的前向传播过程中使用

        '''================================================================================================'''

    def forward(self, f):
        bert_output = f
        offset = self.offset_conv(f)# [4,30,6,7]计算偏移量，使用 offset_conv 对输入 f 进行卷积计算
        offset = self.bn(offset)# 对偏移量进行批归一化
        
        offset = torch.tanh(offset)# 对偏移量进行 tanh 激活函数处理，将其限制在 -1 到 1 的范围内,We need a range of deformation between -1 and 1 to mimic the snake's swing,我们需要在-1到1之间的变形范围来模仿蛇的摆动
        input_shape = f.shape #[4,5,6,7]
        # print(input_shape)
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)#创建 DSC 对象，用于进行可变形卷积操作
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)# 进行可变形卷积操作，得到变形后的特征图deformed_feature

        if self.morph == 0:# 沿着 x 轴进行动态蛇形卷积
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)# 进行组归一化
            x = self.relu(x)# 进行 ReLU 激活函数处理
            '''========================================================================================================'''

            maxPool = nn.MaxPool2d(kernel_size=(12,768))
            #把12层 hidden state 
            x  = 0.99*x + 0.01*bert_output
            x = maxPool(x)#[16,6,1,1]
            x= torch.reshape(x, [-1, 6])  # [16, 6] 
            x = self.Weight(x) + self.bias  # [batch_size, class_num]

            '''========================================================================================================'''

            return x
        
        else:# 沿着 y 轴进行动态蛇形卷积
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)# 进行组归一化
            x = self.relu(x)# 进行 ReLU 激活函数处理

            '''========================================================================================================'''

            maxPool = nn.MaxPool2d(kernel_size=(12,768))
            x = maxPool(x)#[16,6,1,1]
            x= torch.reshape(x, [-1, 6])  # [batch_size, 6] = [16,6]

            x = self.Weight(x) + self.bias  # [batch_size, class_num]

            '''========================================================================================================'''
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]#图象的宽
        self.height = input_shape[3]#图象的高
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        [B,C,W,H]=[4,5,6,7]
        """
        self.num_batch = input_shape[0]#获取batch_size
        self.num_channels = input_shape[1]#获取channels

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>),偏移量的形状[B,2*K,W,H]，因为有两个方向（x,y）需要偏移
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # 得到了
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)#将offset在1维度上进行切分，每个张量在1维度上的大小为self.num_points，就是kenerel_size的大小15
        #y_offset[batch_size,kenerel_size,W,H]=[4,15,6,7] ; x_offset[batch_size,kenerel_size,W,H]=[4,15,6,7] 

        y_center = torch.arange(0, self.width).repeat([self.height])#创建一个张量y_center,包括区间[0，self.width]=[0,6]的整数,并且重复self.height=7次，这时候形状是[42]
        y_center = y_center.reshape(self.height, self.width)#塑形 张量y_center,变成 [self.height, self.width]=[7,6]
        # print(y_center)
        y_center = y_center.permute(1, 0)#第二个维度变成第一个维度 [6,7]: 原来[7,6]的第i列变成第i行
        # print(y_center)
        y_center = y_center.reshape([-1, self.width, self.height])#[1,6,7]增加一个维度
        # print(y_center)
        y_center = y_center.repeat([self.num_points, 1, 1]).float()#[15,6,7]将上一步[1,6,7] 形状的y_center在0维度上重复15次
        # print(y_center)
        y_center = y_center.unsqueeze(0)#[1,15,6,7]增加一个维度
        # print('y_center','\n',y_center)


        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)#[1,15,6,7]

        if self.morph == 0:#0表示在x方向上进行操作
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size) -7~7
                
            """
            y = torch.linspace(0, 0, 1)#[0] 生成了一个包含1个元素的张量 y。这个张量中的所有元素的值都被指定为0
            x = torch.linspace(-int(self.num_points // 2),int(self.num_points // 2),int(self.num_points),)
            #[15]创建了一个包含 self.num_points =15个元素的张量 x，这些元素等间距地分布在从 -num_points // 2 到 num_points // 2 的范围内

            y, x = torch.meshgrid(y, x)
            #调用torch.meshgrid(y, x)将生成一个二维网格，其中y对应于网格的行，x对应于网格的列。1行15列
            #如果y有m个元素，x有n个元素，那么结果将是两个张量，一个形状为(m, n)的y张量，一个形状为(m, n)的x张量。网格的行数等于y的元素个数，列数等于x的元素个数
            y_spread = y.reshape(-1, 1)#[15,1]塑形,重塑为一个一维的 列 向量
            x_spread = x.reshape(-1, 1)#[15,1]塑形,重塑为一个一维的 列 向量



            y_grid = y_spread.repeat([1, self.width * self.height])
            #[15,6*7]调用 y_spread.repeat([1, self.width * self.height]) 在第一个维度上复制一次,在第二个维度上复制self.width * self.height=42次。
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])#[15,6,7] 重塑维度
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]     [1,15,6,7]

            x_grid = x_spread.repeat([1, self.width * self.height])#[15,6*7]
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])#[15,6,7] 
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H] [1,15,6,7]
            # print('y_grid','\n',y_grid)

            y_new = y_center + y_grid
            # print('y_new','\n',y_new)

            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)#[batch_size,15,6,7]
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()#代码的作用是创建了一个名为 y_offset_new 的新张量，它是 y_offset 的一个副本，并且与计算图断开连接。这样可以在后续的计算中使用 y_offset_new 而不会影响到梯度的传播，并且保持 y_offset 原始张量的不变性。

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)#[4,15,6,7]-->[15,4,6,7] 维度重置
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)#[4,15,6,7]-->[15,4,6,7]
                # print('y_offset_new',y_offset_new)
                center = int(self.num_points // 2)#15/2=7

                # The center position remains unchanged and the rest of the positions begin to swing 中心位置保持不变，其余位置开始摆动
                # This part is quite simple. The main idea is that "offset is an iterative process"
                # print(y_offset_new[center])
                y_offset_new[center] = 0#将 y_offset_new 张量中位于 center 位置的元素的值设置为 0 ；y_offset_new形状[15,4,6,7]，有15个形状[4,6,7]的块，让第center=7块的所有元素值等于0
                # print(y_offset_new[center])
                # print('y_offset_new',y_offset_new)
            

                for index in range(1, center):#(1,7)包括1不包括7
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])

                    #y_offset_new[8] = (y_offset_new[7] + y_offset[8])
                    #y_offset_new[9] = (y_offset_new[8] + y_offset[9])

                    #y_offset_new[6] = (y_offset_new[7] + y_offset[6])
                    #y_offset_new[5] = (y_offset_new[6] + y_offset[5])
                    
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)#[4,15,6,7]交换0,1维度
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))#[4,15,6,7]将 y_offset_new 张量乘以扩展因子 self.extend_scope，然后将结果与 y_new 张量进行逐元素相加



            y_new = y_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height])
            
            y_new = y_new.permute(0, 3, 1, 4, 2) #[4,6,15,7,1]
            
            y_new = y_new.reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
            


            x_new = x_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height])
            
            x_new = x_new.permute(0, 3, 1, 4, 2)
            
            x_new = x_new.reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
            
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1 #5
        max_x = self.height - 1 #6

        # find 8 grid locations 
        y0 = torch.floor(y).int() #向下取整
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y) #将y0的每个元素都限制在zero和,max_y之间
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
            # print(outputs)
            # print(outputs.shape)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
            
        return outputs

    def deform_conv(self, 
                    input, #输入f
                    offset, #偏移量
                    if_offset#是否偏移，不偏移就是普通卷积
                    ):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature #[4,5,90,7]


if __name__ == '__main__':

    '''raw====================================================================================='''
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # A = np.random.rand(4, 5, 6, 7)#(batch_size,in_ch,width,height)
    # A = np.random.rand(4, 5, 6, 7)#(batch_size,in_ch,width,height)           
    # # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
    # # print(A)
    # A = A.astype(dtype=np.float32)
    # A = torch.from_numpy(A)
    # print(A.shape)

    # conv0 = SaDCNConv(
    #     in_ch=5,
    #     out_ch=6,
    #     kernel_size=15,#卷积核 15*15
    #     extend_scope=1,
    #     morph=0,
    #     if_offset=True,
    #     device=device)
    
    # if torch.cuda.is_available():
    #     A = A.to(device)
    #     conv0 = conv0.to(device)
    # out = conv0(A)
    # # print(out)
    # print(out.shape)



    '''text====================================================================================='''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A = np.random.rand(4, 5, 6, 7)#(batch_size,in_ch,width,height)
    A = np.random.rand(16, 5, 12, 768)#(batch_size,in_ch,width,height)           (16,5,12,768)

    # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
    # print(A)
    A = A.astype(dtype=np.float32)
    A = torch.from_numpy(A)
    print('input_shape',A.shape)

    conv0 = SaDCNConv(
        in_ch=5,
        out_ch=6,
        kernel_size=15,#卷积核 15*15
        extend_scope=1,
        morph=0,
        if_offset=True,
        device=device)
    
    if torch.cuda.is_available():
        A = A.to(device)
        conv0 = conv0.to(device)
    out = conv0(A)
    # print(out)
    print('output_shape',out.shape)