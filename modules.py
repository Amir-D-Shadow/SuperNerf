import torch
import numpy as np
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self,in_dim,hidden_dim,kernel_size,stride=1,padding = 0):

        super(ConvLSTMCell,self).__init__()

        """
        param in_dim: Number of channels of input tensor
        param hidden_dim: Number of channels of hidden state
        """

        self.hidden_dim = hidden_dim


        self.conv = nn.Conv2d(in_channels=in_dim+hidden_dim,
                              out_channels = 4 * hidden_dim,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding
                              )

        self.norm = nn.LayerNorm(4 * hidden_dim,eps=1e-6)

    def forward(self,x,prev_state):

        """
        params x : Tensor (N,C,H,W)
        params prev_state : list [h_prev,c_prev]
        h_prev : (N,C,H,W)
        c_prev : (N,C,H,W)
        """

        h_prev,c_prev = prev_state

        combined = torch.cat([x,h_prev],dim=1) #(N,C+hidden_dim,H,W)

        combined_conv = self.conv(combined)  #(N,4*hidden_dim,H,W)

        combined_conv = combined_conv.permute(0,2,3,1) #(N,H,W,4*hidden_dim)

        combined_conv = self.norm(combined_conv) #(N,H,W,4*hidden_dim)

        combined_conv = combined_conv.permute(0,3,1,2) #(N,4*hidden_dim,H,W)
 
        #cc_i: (N,hidden_dim,H,W), cc_f: (N,hidden_dim,H,W),cc_o: (N,hidden_dim,H,W),cc_g: (N,hidden_dim,H,W)
        cc_i,cc_f,cc_o,cc_g = torch.split(combined_conv,self.hidden_dim,dim=1)

        c_cur = torch.tanh(cc_g) #(N,hidden_dim,H,W)
        
        i = torch.sigmoid(cc_i) #(N,hidden_dim,H,W)
        f = torch.sigmoid(cc_f) #(N,hidden_dim,H,W)
        o = torch.sigmoid(cc_o) #(N,hidden_dim,H,W)

        c_next = i * c_cur + f * c_prev #(N,hidden_dim,H,W)
        h_next = o * torch.tanh(c_next) #(N,hidden_dim,H,W)

        return h_next,c_next
        

class ConvLSTM(nn.Module):

    def __init__(self,in_dim,hidden_dim,kernel_size,stride=1,padding = 0):

        """
        params size_of_state : tuple (N,C,H,W)
        """
        super(ConvLSTM,self).__init__()

        self.LSTM_Cell = ConvLSTMCell(in_dim=in_dim,
                                      hidden_dim = hidden_dim,
                                      kernel_size=kernel_size,
                                      stride = stride,
                                      padding = padding
                                      )

        self.init_h = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1) #(N,C,H,W) C = hidden_dim
        self.init_c = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1) #(N,C,H,W) C = hidden_dim


    def forward(self,x):

        """
        param x : tensor (N,T,C,H,W)
        param first_state_flag : bool , if the first state
        """

        seq_len = x.shape[1]

        h = self.init_h(x[:,0,:,:,:]) #(N,C,H,W)
        c = self.init_c(x[:,0,:,:,:]) #(N,C,H,W)

        layer_out = []

        for t in range(seq_len):

            h,c = self.LSTM_Cell(x[:,t,:,:,:],prev_state=[h,c])

            layer_out.append(h)


        layer_out = torch.stack(layer_out,dim=1) #(N,T,C,H,W)
        

        return layer_out


class BiConvLSTM(nn.Module):

    def __init__(self,in_dim,hidden_dim,kernel_size,stride=1,padding = 0):

        super(BiConvLSTM,self).__init__()

        self.forwardNet = ConvLSTM(in_dim = in_dim,
                                   hidden_dim = hidden_dim,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = padding
                                   )

        self.reverseNet = ConvLSTM(in_dim = in_dim,
                                    hidden_dim = hidden_dim,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding
                                    )

    def forward(self,xfor,xrev):

        """
        param xfor: Tensor (N,T,C,H,W) 
        param xrev: Tensor (N,T,C,H,W)
        """

        yfor = self.forwardNet(xfor) #(N,T,C,H,W)
        yrev = self.reverseNet(xrev) #(N,T,C,H,W)

        reversed_idx = list(reversed(range(yrev.shape[1])))
        yrev = yrev[:,reversed_idx,:,:,:] #(N,T,C,H,W)

        yout = torch.cat([yfor,yrev],dim=2) #(N,T,2*C,H,W) , C = hidden_dim
        
        return yout
        

class DWConv(nn.Module):

    def __init__(self,in_dim,out_dim,kernel_size,stride=1,padding=0):

        super(DWConv,self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_dim,
                                   out_channels=in_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_dim
                                   )
        
        self.pointwise = nn.Conv2d(in_channels=in_dim,
                                   out_channels=out_dim,
                                   kernel_size=1
                                   )


    def forward(self,x):

        #x: (N,C,H,W)

        y = self.depthwise(x)
        y = self.pointwise(y)

        return y

              
class DWConvNeX(nn.Module):

    def __init__(self,in_dim,D,kernel_size,stride=1,padding=0):

        super(DWConvNeX,self).__init__()

        self.conv0 = DWConv(in_dim=in_dim,
                            out_dim=D,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                            )

        self.norm0 = nn.LayerNorm(D,eps=1e-6)

        self.act0 = nn.ReLU()

        self.proj_layer = nn.Linear(in_features=D,out_features=in_dim)


    def forward(self,x):

        #x : (N,in_dim,H,W)

        y = self.conv0(x) #(N,D,H,W)

        y = y.permute(0,2,3,1) #(N,H,W,D)

        y = self.norm0(y) #(N,H,W,D)

        y = self.act0(y) #(N,H,W,D)

        y = self.proj_layer(y) #(N,H,W,in_dim)

        y = y.permute(0,3,1,2) #(N,in_dim,H,W)

        out = y + x #(N,in_dim,H,W)

        return out

class FeatTempInp(nn.Module):

    def __init__(self,in_dim,out_dim,D):

        super(FeatTempInp,self).__init__()

        self.conv_sample = nn.Conv3d(in_channels=in_dim,
                                     out_channels=D,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1
                                     )

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.blending_Tminus1 = nn.Conv3d(in_channels=D,
                                          out_channels=out_dim,
                                          kernel_size=1,
                                          stride=1
                                          )

        self.blending_Tplus1 =  nn.Conv3d(in_channels=D,
                                          out_channels=out_dim,
                                          kernel_size=1,
                                          stride=1
                                          )

    def forward(self,x1,x3):

        """
        params x1 : Tensor (N,C,N_sample,H,W)
        params x3 : Tensor (N,C,N_sample,H,W)
        """

        y1 = self.conv_sample(x1) #(N,C,N_sample,H,W)
        y1 = self.act1(y1)         #(N,C,N_sample,H,W)
        y1 = self.blending_Tminus1(y1)  #(N,C,N_sample,H,W)
        
        y3 = self.conv_sample(x3) #(N,C,N_sample,H,W)
        y3 = self.act2(y3)        #(N,C,N_sample,H,W)
        y3 = self.blending_Tplus1(y3) #(N,C,N_sample,H,W)

        y2 = y1 + y3 #(N,C,N_sample,H,W)

        return y2

        
        
        

        
#utils
def setTrainState(model,train_state=True):

    """
    param model : pytorch model that required to change the state of require grad
    param train_state : bool 
    """

    for p in model.parameters():

        p.requires_grad = train_state








    
