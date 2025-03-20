import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from pydub import AudioSegment
import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 项目根目录路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_audio_segment(audio_path):
    """提取音频片段"""
    print(f"正在加载音频文件: {audio_path}")
    
    if audio_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
        # 临时保存为wav格式
        temp_wav = os.path.join(ROOT_DIR, "temp_segment.wav")
        audio.export(temp_wav, format="wav")
        
        # 使用librosa加载
        y, sr = librosa.load(temp_wav, sr=None)
        
        # 删除临时文件
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    else:
        y, sr = librosa.load(audio_path, sr=None)
    
    print(f"音频加载成功! 采样率: {sr}Hz, 长度: {len(y)/sr:.2f}秒")
    return y, sr

def compute_mel_spectrogram(y, sr, n_mels=128):
    """计算梅尔频谱图"""
    # 计算STFT
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(D)
    
    # 转换为梅尔频谱图，确保覆盖0-20kHz范围
    mel_spec = librosa.feature.melspectrogram(
        S=magnitude,
        sr=sr,
        n_mels=n_mels,
        fmin=0,
        fmax=20000
    )
    
    # 转换为分贝单位
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def get_mel_filter_width(center_freq, base_width):
    """根据梅尔频率计算滤波器宽度
    :param center_freq: 中心频率(Hz)
    :param base_width: 基准宽度
    :return: 对应梅尔刻度的滤波器宽度
    """
    # 将中心频率转换为梅尔刻度
    mel_center = librosa.hz_to_mel(center_freq)
    mel_base = librosa.hz_to_mel(200)  # 基准频率的梅尔值
    
    # 计算相对于基准频率的梅尔刻度比例
    mel_ratio = mel_center / mel_base
    
    # 调整滤波器宽度，确保随频率增加而适当增加
    width = int(base_width * mel_ratio)
    
    # 设置最小和最大限制
    return max(3, min(width, int(base_width * 1.5)))  # 限制最大宽度为基准宽度的1.5倍

def apply_mel_filter(mel_spec, sr, filter_type='max', time_width=3, freq_base_width=5, n_mels=128):
    """应用基于梅尔刻度的滤波器
    :param mel_spec: 输入的梅尔频谱图
    :param sr: 采样率
    :param filter_type: 滤波器类型，可选 'max', 'min', 'median'
    :param time_width: 时间轴上的窗口宽度
    :param freq_base_width: 频率轴上的基准窗口宽度
    :param n_mels: 梅尔频带数量
    """
    # 获取梅尔频率数组
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=20000)
    
    # 创建输出数组
    filtered_spec = np.zeros_like(mel_spec)
    
    # 选择滤波器类型
    if filter_type == 'max':
        filter_func = ndimage.maximum_filter
    elif filter_type == 'min':
        filter_func = ndimage.minimum_filter
    else:  # median
        filter_func = ndimage.median_filter
    
    # 对每个频率点使用不同大小的滤波器
    for i, freq in enumerate(mel_freqs):
        # 获取当前频率的滤波器宽度
        freq_width = get_mel_filter_width(freq, freq_base_width)
        
        # 创建当前频率的滤波器核
        kernel = np.ones((freq_width, time_width))
        
        # 提取当前频率附近的区域进行滤波
        start_freq = max(0, i - freq_width // 2)
        end_freq = min(n_mels, i + freq_width // 2 + 1)
        
        # 应用滤波器
        local_region = mel_spec[start_freq:end_freq, :]
        if local_region.size > 0:  # 确保有数据可处理
            if filter_type == 'median':
                # 对于中值滤波，我们需要特别处理边界
                # 使用mode='reflect'来处理边界，这样可以避免边缘效应
                filtered_region = ndimage.median_filter(
                    local_region,
                    size=(freq_width, time_width),
                    mode='reflect'
                )
            else:
                filtered_region = filter_func(local_region, footprint=kernel)
            
            # 确保我们只取中心行的结果
            center_idx = min(freq_width // 2, filtered_region.shape[0] // 2)
            filtered_spec[i, :] = filtered_region[center_idx]
    
    return filtered_spec

def plot_spectrograms_grid(original_spec, filtered_specs_dict, sr, filter_type, save_path=None):
    """绘制原始频谱图和不同参数组合的滤波结果"""
    # 将结果分成多个图，每个图最多显示19个滤波结果（加上原始图共20个）
    max_subplots_per_figure = 19
    n_combinations = len(filtered_specs_dict)
    n_figures = (n_combinations + max_subplots_per_figure - 1) // max_subplots_per_figure
    
    # 将参数组合分成多组
    param_groups = []
    items = list(filtered_specs_dict.items())
    for i in range(n_figures):
        start_idx = i * max_subplots_per_figure
        end_idx = min((i + 1) * max_subplots_per_figure, n_combinations)
        param_groups.append(items[start_idx:end_idx])
    
    # 为每组参数创建一个图
    for fig_idx, param_group in enumerate(param_groups):
        n_subplots = len(param_group) + 1  # +1 for original spectrogram
        n_rows = (n_subplots + 4) // 5  # 每行5个子图
        
        plt.figure(figsize=(25, 5 * n_rows))
        
        # 绘制原始梅尔频谱图
        plt.subplot(n_rows, 5, 1)
        librosa.display.specshow(
            original_spec,
            y_axis='mel',
            x_axis='time',
            sr=sr,
            fmin=0,
            fmax=20000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('原始梅尔频谱图 (0-20kHz)')
        
        # 绘制当前组的滤波结果
        for idx, (params, spec) in enumerate(param_group, 2):
            time_width, freq_width = params
            plt.subplot(n_rows, 5, idx)
            librosa.display.specshow(
                spec,
                y_axis='mel',
                x_axis='time',
                sr=sr,
                fmin=0,
                fmax=20000
            )
            plt.colorbar(format='%+2.0f dB')
            filter_name = {
                'max': '最大值',
                'min': '最小值',
                'median': '中值'
            }[filter_type]
            plt.title(f'{filter_name}滤波\n时间={time_width}, 频率={freq_width}')
        
        # 调整布局并保存图像
        plt.tight_layout()
        
        if save_path:
            # 为多个图像添加编号
            base, ext = os.path.splitext(save_path)
            current_save_path = f"{base}_{fig_idx + 1}{ext}" if n_figures > 1 else save_path
            plt.savefig(current_save_path, dpi=300)
            print(f"频谱图已保存到: {current_save_path}")
        
        plt.close()

def generate_parameter_combinations(n_combinations=18):
    """生成时间宽度和频率宽度的组合
    :param n_combinations: 需要生成的组合数量
    :return: 参数组合列表 [(time_width, freq_width), ...]
    """
    # 生成基础参数范围
    time_widths = np.linspace(3, 21, 7).astype(int)  # [3, 6, 9, 12, 15, 18, 21]
    freq_widths = np.linspace(3, 21, 7).astype(int)  # [3, 6, 9, 12, 15, 18, 21]
    
    # 生成所有可能的组合
    all_combinations = []
    for t in time_widths:
        for f in freq_widths:
            # 确保时间宽度和频率宽度的比例在0.5到2之间
            if 0.5 <= t/f <= 2:
                all_combinations.append((t, f))
    
    # 如果组合数量不够，添加一些中间值
    if len(all_combinations) < n_combinations:
        extra_widths = np.linspace(4, 20, 5).astype(int)  # [4, 8, 12, 16, 20]
        for w in extra_widths:
            all_combinations.append((w, w))  # 添加一些相等的宽度组合
    
    # 随机选择指定数量的组合
    np.random.seed(42)  # 设置随机种子以确保可重复性
    selected_indices = np.random.choice(len(all_combinations), size=min(n_combinations, len(all_combinations)), replace=False)
    selected_combinations = [all_combinations[i] for i in selected_indices]
    
    return selected_combinations

if __name__ == "__main__":
    # 创建带时间戳的主输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(ROOT_DIR, f"滤波器结果_{timestamp}")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    # 创建三个子文件夹
    filter_types = {
        'max': '最大值滤波',
        'min': '最小值滤波',
        'median': '中值滤波'
    }
    
    output_dirs = {}
    for filter_type, folder_name in filter_types.items():
        output_dirs[filter_type] = os.path.join(main_output_dir, folder_name)
        if not os.path.exists(output_dirs[filter_type]):
            os.makedirs(output_dirs[filter_type])
    
    # 只处理原始音频片段.mp3文件
    print("\n===== 处理原始音频片段.mp3 =====")
    snippet_path = os.path.join(ROOT_DIR, "原始音频片段.mp3")
    
    if os.path.exists(snippet_path):
        # 加载音频
        y, sr = extract_audio_segment(snippet_path)
        
        # 计算梅尔频谱图
        mel_spec = compute_mel_spectrogram(y, sr)
        
        # 生成参数组合
        param_combinations = generate_parameter_combinations(18)
        
        # 对每种滤波类型进行处理
        for filter_type in filter_types.keys():
            print(f"\n正在处理{filter_types[filter_type]}...")
            
            # 存储不同参数组合的滤波结果
            filtered_specs = {}
            
            # 应用不同参数组合的滤波器
            for t_width, f_width in param_combinations:
                print(f"  处理参数组合: 时间宽度={t_width}, 频率宽度={f_width}")
                filtered_specs[(t_width, f_width)] = apply_mel_filter(
                    mel_spec, sr, 
                    filter_type=filter_type,
                    time_width=t_width,
                    freq_base_width=f_width
                )
            
            # 绘制并比较频谱图
            plot_spectrograms_grid(
                mel_spec,
                filtered_specs,
                sr,
                filter_type,
                save_path=os.path.join(output_dirs[filter_type], f"原始音频片段_{filter_types[filter_type]}_渐进对比.png")
            )
            
            print(f"{filter_types[filter_type]}处理完成")
    else:
        print(f"错误: 未找到文件 {snippet_path}")
    
    print(f"\n所有处理完成，结果已保存到: {main_output_dir}")
    print("子文件夹:")
    for folder_name in filter_types.values():
        print(f"- {folder_name}")
    print("\n参数组合:")
    for t_width, f_width in param_combinations:
        print(f"时间宽度: {t_width}, 频率宽度: {f_width}")
