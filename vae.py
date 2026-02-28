import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, os
if os.environ.get("NO_PDB", "0") == "1":
    pdb.set_trace = lambda: None


import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def shared_eval(self, batch, optimizer, mode, comet_logger='None'):
        pass

    def configure_optimizers(self, lr=1e-3):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # adds weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer


# activation_function_name = 'silu' # 'relu' or 'silu'
# if activation_function_name == 'relu':
#     activation_function = F.relu
# elif activation_function_name == 'silu':
#     activation_function = F.silu
# else:
#     raise NotImplementedError

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, activation_function_name):
        super(Residual, self).__init__()
        if activation_function_name == 'relu':
            act_fn1 = nn.ReLU(True)  # 改用更清晰的命名
            act_fn2 = nn.ReLU(True)
        elif activation_function_name == 'silu':
            act_fn1 = nn.SiLU(True)
            act_fn2 = nn.SiLU(True)

        self._block = nn.Sequential(
            act_fn1,
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False), # 保持 seq_len 不变
            act_fn2,
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False) # 保持 seq_len 不变
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, activation_function_name, activation_function):
        super(ResidualStack, self).__init__()
        self.activation_function = activation_function
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, activation_function_name)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return self.activation_function(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor, activation_function_name, activation_function):
        super(Encoder, self).__init__()
        self.activation_function = activation_function
        if compression_factor == 4: # in_channels=1 num_hiddens=128, num_residual_layers=2, num_residual_hiddens=64, embedding_dim=64, compression_factor=4
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1) # 使得 seq_len 减半
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1) # 使得 seq_len 再减半. 于是成为原本的1/4
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1) # 保持 seq_len 不变
            self._residual_stack = ResidualStack(in_channels=num_hiddens, # 这是带残差连接的卷积
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1) # 本质上是在通道轴的线性投影

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = self.activation_function(x)

            x = self._conv_2(x)
            x = self.activation_function(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x # (batch, embedding_dim, seq_len/compression_factor)

        elif compression_factor == 8:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = self.activation_function(x)

            x = self._conv_2(x)
            x = self.activation_function(x)

            x = self._conv_A(x)
            x = self.activation_function(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = self.activation_function(x)

            x = self._conv_2(x)
            x = self.activation_function(x)

            x = self._conv_3(x)
            x = self.activation_function(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = self.activation_function(x)

            x = self._conv_2(x)
            x = self.activation_function(x)

            x = self._conv_A(x)
            x = self.activation_function(x)

            x = self._conv_B(x)
            x = self.activation_function(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor, activation_function_name, activation_function):
        super(Decoder, self).__init__()
        self.activation_function = activation_function
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1) # seq_len 不变

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            # To get the correct shape back the kernel size has to be 5 not 4
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=5,
                                                    stride=3, padding=1)

            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens,
                                                 activation_function_name=activation_function_name, activation_function=activation_function,
                                                 )

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_B = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_1(x)
            x = self.activation_function(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 8:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = self.activation_function(x)

            x = self._conv_trans_1(x)
            x = self.activation_function(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 12:
            x = self._conv_1(inputs)
            x = self._residual_stack(x)

            x = self._conv_trans_2(x)
            x = self.activation_function(x)

            x = self._conv_trans_3(x)
            x = self.activation_function(x)

            x = self._conv_trans_4(x)

            return torch.squeeze(x)

        elif compression_factor == 16:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = self.activation_function(x)

            x = self._conv_trans_B(x)
            x = self.activation_function(x)

            x = self._conv_trans_1(x)
            x = self.activation_function(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous() # (batch,embedding_dim, T/F) -> (batch, T/F, embedding_dim)
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim) # (batch*T/F, embedding_dim)

        # Calculate distances (使用了 (a-b)^2 = a^2 + b^2 - 2ab 的技巧)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (batch*T/F, 1) 这一步无法导数传递, 因为是离散的
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1) # (batch*T/F, num_embeddings), one-hot 编码

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight # (batch*T/F, embedding_dim)
                                 ).view(input_shape) # (batch, T/F, embedding_dim)
        # 历史遗留
        # 这段代码源自 DeepMind 早期的 VQ-VAE 实现。在早期版本的 TensorFlow 和 PyTorch 中，索引操作的梯度支持并不完善，使用 one-hot + matmul 是更"安全"的做法。                

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # 把 quantized 当作常数, 只更新 inputs, 目的是让 inputs(编码器输出) 靠近 quantized(码字)
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # 把 inputs 当作常数, 只更新 quantized, 目的是让 quantized(码字) 靠近 inputs(编码器输出)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss # 总的 loss, 一般 commitment_cost 取 0.25, 相当于更倾向于让 quantized(码字) 靠近 inputs(编码器输出)

        quantized = inputs + (quantized - inputs).detach() # 通过这种方式让梯度直接传递给 inputs(编码器输出), 而不更新 quantized(码字), 这样就实现了 "straight-through" gradient estimator

        avg_probs = torch.mean(encodings, dim=0) # (num_embeddings,), 计算每个码字被使用的频率(1个batch*T/F 内)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # 计算 perplexity, 衡量码字的使用情况, 越接近 num_embeddings 越好
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings
                            #(batch, embedding_dim, T/F)                                          (batch*T/F, 1)   (batch*T/F, num_embeddings)

class vqvae(BaseModel):
    loss_type = 'mse' # mse or l1loss
    def __init__(self, vqvae_config):
        super().__init__()
        num_hiddens = vqvae_config['block_hidden_size']
        num_residual_layers = vqvae_config['num_residual_layers']
        num_residual_hiddens = vqvae_config['res_hidden_size']
        embedding_dim = vqvae_config['embedding_dim']
        num_embeddings = vqvae_config['num_embeddings']
        commitment_cost = vqvae_config['commitment_cost']
        self.compression_factor = vqvae_config['compression_factor']
        if 'loss_type' in vqvae_config:
            self.loss_type = vqvae_config['loss_type']
            print(f"loss_type: {self.loss_type}")
        else:
            print("loss_type not found in vqvae_config")

        if 'activation_function' in vqvae_config:
            self.activation_function_name = vqvae_config['activation_function']
            print(f"activation_function_name: {self.activation_function_name}")
        else:
            self.activation_function_name = 'relu'
            print(f"activation_function not found in vqvae_config, use {self.activation_function_name} default")
        self.activation_function = self.get_activation_function(self.activation_function_name)


        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.encoder = Encoder(1,             num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, self.compression_factor, self.activation_function_name, self.activation_function)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, self.compression_factor, self.activation_function_name, self.activation_function)

    def get_activation_function(self, activation_function_name):
        if activation_function_name == 'relu':
            activation_function = F.relu
        elif activation_function_name == 'silu':
            activation_function = F.silu
        else:
            raise NotImplementedError
        return activation_function

    # 【新增】标准的 forward 方法，用于 DataParallel
    def forward(self, batch):
        """返回loss和所有需要的中间结果
        输入: batch, shape: (batch, seq_len)
        """
        z = self.encoder(batch, self.compression_factor)
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
        data_recon = self.decoder(quantized, self.compression_factor)

        if data_recon.ndim == 1: data_recon = data_recon.unsqueeze(0)
        
        if self.loss_type == 'mse':
            recon_error = F.mse_loss(data_recon, batch)
        elif self.loss_type == 'l1':
            recon_error = F.l1_loss(data_recon, batch)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        loss = recon_error + vq_loss
        recon_error = F.mse_loss(data_recon, batch) # 用于显示指标
        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings


    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        if mode == 'train':
            optimizer.zero_grad()
            z = self.encoder(batch, self.compression_factor) # (batch, embedding_dim, seq_len/compression_factor)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized, self.compression_factor)
            if self.loss_type == '':
                pdb.set_trace()
            if self.loss_type == 'mse':
                recon_error = F.mse_loss(data_recon, batch)
            elif self.loss_type == 'l1':
                recon_error = F.l1_loss(data_recon, batch) # 增加训练稳定性，防止码本坍塌
            loss = recon_error + vq_loss
            loss.backward()
            recon_error = F.mse_loss(data_recon, batch) # 用于显示指标
            optimizer.step()

        if mode == 'val' or mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch, self.compression_factor)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
                data_recon = self.decoder(quantized, self.compression_factor)
                recon_error = F.mse_loss(data_recon, batch)
                loss = recon_error + vq_loss

        comet_logger.log_metric(f'{mode}_vqvae_loss_each_batch', loss.item())
        comet_logger.log_metric(f'{mode}_vqvae_vq_loss_each_batch', vq_loss.item())
        comet_logger.log_metric(f'{mode}_vqvae_recon_loss_each_batch', recon_error.item())
        comet_logger.log_metric(f'{mode}_vqvae_perplexity_each_batch', perplexity.item())

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight,                        encoding_indices,                 encodings
        #                                                        码本: (num_embeddings, embedding_dim)  编码索引: (batch*T/F, 1)  编码 one-hot: (batch*T/F, num_embeddings)

if __name__ == "__main__":
    model_path = ""
    