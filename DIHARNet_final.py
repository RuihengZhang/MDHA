import math
import time
import torch
import torch.nn as nn
from thop import profile
from einops import rearrange
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fghar_p import CNN_encoder
from models.fghar_m import Channel_MobileViTBlock, Down_wt


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        # self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(128, emb_dim)
        self.Wv = nn.Linear(128, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # context shape: [3, 1, 256]
        b, c, h, w = x.shape

        x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 128, 8, 8]
        x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] =  [3, 64, 128]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] =  [3, 64, 128]
        # print('Q:', Q.shape)
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 1, 128]
        # print('K:', K.shape)
        V = self.Wv(context)  # [batch_szie, seq_len, emb_dim] = [3, 1, 128]
        # print('V:', V.shape)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)  # [3, 1, 64]

        att_weights = att_weights * self.scale
        # print('att_weights.shape', att_weights.shape)

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)  # [3, 64, 1]
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim] = [3, 64, 128]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w] = [3, 128, 8, 8]
        out = self.proj_out(out)   # [batch_size, c, h, w]
        # print('after attention:', out.shape)

        # print(out.shape)

        return out, att_weights


class FGHAR(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [1, 1, 3]
        self.conv1 = conv_nxn_bn(1, channels[0], stride=2)

        self.dwt_1 = Down_wt(channels[1], channels[2])
        self.dwt_2 = Down_wt(channels[3], channels[4])

        self.vst_encoder = VST_encoder(channels, num_classes, expansion=2)

        self.ca = CrossAttention(48, 128)
        self.cmvit_1 = Channel_MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2))

        self.dwt_3 = Down_wt(channels[5], channels[6])
        self.cmvit_2 = Channel_MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[0] * 2))

        self.dwt_4 = Down_wt(channels[7], channels[8])
        self.cmvit_3 = Channel_MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[0] * 2))
        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)


    def forward(self, md, v, timing=False):
        if timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
        
        md = self.conv1(md)

        md = self.dwt_1(md)
        md_in = self.dwt_2(md)  # (16, 48, 8, 8)
        # print('after dwt_2:', md_in.shape)

        v_in = self.vst_encoder(v)
        # print('after vst_encoder:', v_in.shape)
        v_in = v_in.unsqueeze(1)  # [3, 1, 256]
        # print('afrer unsqueeze:', v_in.shape)

        fea, att_map = self.ca(md_in, v_in)
        x = self.cmvit_1(fea)

        x = self.dwt_3(x)
        x = self.cmvit_2(x)

        x = self.dwt_4(x)
        x = self.cmvit_3(x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        
        if timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            return x, inference_time
        
        return x, v_in, md_in, fea, att_map

class VST_encoder(nn.Module):
    def __init__(self, channels, num_classes, expansion=2):
        super().__init__()
        self.cnn_encoder = CNN_encoder(channels, num_classes, expansion)
        self.lstm = nn.LSTM(input_size=150, hidden_size=128, num_layers=1)

    def forward(self, x):
        hidden = None
        for t in range(x.size(1)):
            fea = self.cnn_encoder(x[:, t, :, :, :])
            out, hidden = self.lstm(fea.unsqueeze(0), hidden)
        return out[-1, :, :]

def fghar():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return FGHAR((64, 64), dims, channels, num_classes=11, expansion=2)

def measure_inference_time(model, md_input, v_input, num_runs=100, warmup_runs=10):
    """
    Measure the inference time of the model with multiple runs for statistical accuracy.
    
    Args:
        model: The FGHAR model
        md_input: Micro-Doppler input tensor
        v_input: Voxel input tensor 
        num_runs: Number of inference runs for timing measurement
        warmup_runs: Number of warmup runs before timing
        
    Returns:
        dict: Dictionary containing timing statistics
    """
    model.eval()
    device = next(model.parameters()).device
    md_input = md_input.to(device)
    v_input = v_input.to(device)
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(md_input, v_input)
    
    # Timing runs
    print(f"Running {num_runs} timing iterations...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(md_input, v_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} runs")
    
    # Calculate statistics
    import statistics
    avg_time = statistics.mean(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
    median_time = statistics.median(inference_times)
    
    # Calculate FPS (Frames Per Second)
    fps = 1000.0 / avg_time if avg_time > 0 else 0
    
    timing_stats = {
        'average_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time,
        'median_time_ms': median_time,
        'fps': fps,
        'num_runs': num_runs,
        'device': str(device)
    }
    
    print("\n" + "="*50)
    print("INFERENCE TIME STATISTICS")
    print("="*50)
    print(f"Device: {device}")
    print(f"Number of runs: {num_runs}")
    print(f"Average inference time: {avg_time:.3f} ms")
    print(f"Min inference time: {min_time:.3f} ms")
    print(f"Max inference time: {max_time:.3f} ms")
    print(f"Standard deviation: {std_time:.3f} ms")
    print(f"Median inference time: {median_time:.3f} ms")
    print(f"Throughput (FPS): {fps:.2f}")
    print("="*50)
    
    return timing_stats

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample inputs
    md = torch.randn(1, 1, 64, 64)
    v = torch.rand(1, 20, 10, 32, 32)
    
    # Create model and move to device
    model = fghar()
    model = model.to(device)
    md = md.to(device)
    v = v.to(device)
    
    # Test model output
    print("\n" + "="*30)
    print("MODEL OUTPUT TEST")
    print("="*30)
    out = model(md, v)
    print(f"Output shape: {out[0].shape}")
    
    # Test single inference with timing
    print("\n" + "="*30)
    print("SINGLE INFERENCE TIMING")
    print("="*30)
    model.eval()
    with torch.no_grad():
        out_with_timing, single_time = model(md, v, timing=True)
        print(f"Single inference time: {single_time:.3f} ms")
    
    # Compute FLOPs and parameters
    print("\n" + "="*30)
    print("MODEL COMPLEXITY")
    print("="*30)
    flops, params = profile(model, inputs=(md, v))
    gflops = flops / 1e9
    params_mb = (params * 4) / 1e6
    params_k = params / 1e3
    print(f"GFLOPs: {gflops:.3f}")
    print(f"Parameters: {params_k:.1f}K ({params_mb:.2f} MB)")
    
    # Comprehensive timing analysis
    print("\n" + "="*40)
    print("COMPREHENSIVE TIMING ANALYSIS")
    print("="*40)
    timing_stats = measure_inference_time(model, md, v, num_runs=100, warmup_runs=20)
    
    # Summary report
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Model: DIHARNet (FGHAR)")
    print(f"Device: {device}")
    print(f"Input shapes: MD={md.shape}, V={v.shape}")
    print(f"Output shape: {out[0].shape}")
    print(f"Parameters: {params_k:.1f}K ({params_mb:.2f} MB)")
    print(f"FLOPs: {gflops:.3f} G")
    print(f"Average inference time: {timing_stats['average_time_ms']:.3f} ± {timing_stats['std_time_ms']:.3f} ms")
    print(f"Throughput: {timing_stats['fps']:.2f} FPS")
    print("="*50)







