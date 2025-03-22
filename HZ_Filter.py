import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
import datetime
import matplotlib.font_manager as fm
import itertools
from scipy import signal
from scipy.ndimage import binary_dilation
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置matplotlib支持中文
def setup_chinese_font():
    # 尝试设置几种常见的中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'FangSong']
    font_found = False
    
    for font_name in font_list:
        # 查找系统中是否有该字体
        font_path = fm.findfont(fm.FontProperties(family=font_name))
        if os.path.exists(font_path) and font_name in font_path:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            print(f"使用中文字体: {font_name}")
            font_found = True
            break
    
    if not font_found:
        print("未找到中文字体，使用默认字体")
        # 如果找不到中文字体，可以尝试更通用的设置
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()

def create_output_folders():
    """创建输出文件夹结构"""
    # 以日期时间命名主输出文件夹
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(ROOT_DIR, f"HZ滤波实验_负增益处理_{timestamp}")
    
    # 创建主输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建子文件夹
    hz_filter_dir = os.path.join(output_dir, "频率轴微分滤波")
    audio_dir = os.path.join(output_dir, "处理后音频")
    
    for directory in [hz_filter_dir, audio_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    print(f"创建输出文件夹: {output_dir}")
    return output_dir, hz_filter_dir, audio_dir

def load_audio(audio_path, start_sec=None, end_sec=None, target_sr=None):
    """加载音频文件，可以指定目标采样率进行下采样"""
    print(f"正在加载音频: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    
    # 如果没有指定时间范围，默认只加载前30秒
    if start_sec is None:
        start_sec = 0
    if end_sec is None:
        end_sec = 30  # 默认只加载30秒音频，避免内存溢出
    
    print(f"加载音频片段: {start_sec}-{end_sec}秒")
    
    # 加载音频片段，如果指定了目标采样率，则进行下采样
    if target_sr:
        print(f"将进行下采样到 {target_sr}Hz")
        y, sr = librosa.load(audio_path, sr=target_sr, offset=start_sec, duration=(end_sec-start_sec))
    else:
        y, sr = librosa.load(audio_path, sr=None, offset=start_sec, duration=(end_sec-start_sec))
    
    print(f"音频加载成功! 采样率: {sr}Hz, 长度: {len(y)/sr:.2f}秒")
    return y, sr

def save_as_wav(audio, sr, filename, output_dir):
    """保存为wav格式"""
    output_path = os.path.join(output_dir, filename)
    sf.write(output_path, audio, sr)
    print(f"音频已保存到: {output_path}")
    return output_path

def plot_spectrogram(audio, sr, title, filename, output_dir):
    """绘制并保存频谱图"""
    plt.figure(figsize=(10, 6))
    
    # 计算STFT
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 绘制频谱图
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def apply_processing_pipeline(audio, sr, output_dir, threshold_db=-35):
    """应用处理管道：阈值滤波 -> 低频截止 -> 微分滤波 -> 膨胀 -> DBSCAN聚类 -> 负增益处理"""
    # 计算STFT
    n_fft = 1024
    hop_length = n_fft // 4
    
    print(f"进行STFT转换，使用n_fft={n_fft}, hop_length={hop_length}")
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 保存原始STFT以便后续处理
    original_D = D.copy()
    original_magnitude = magnitude.copy()
    
    # 转换为dB尺度用于显示和阈值滤波
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 创建阈值掩码
    threshold_mask = magnitude_db < threshold_db
    
    # 固定参数
    height = 10
    intensity = 0.5
    order = 2
    low_freq_cut = True
    
    print(f"使用固定参数: 高度={height}, 强度={intensity}, 阶数={order}")
    
    # 获取频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 创建3kHz以下的频率掩码（用于低频截止）
    low_freq_mask = freqs < 3000
    
    # 先绘制原始频谱
    plt.figure(figsize=(15, 10))
    
    # 原始频谱
    plt.subplot(2, 2, 1)
    librosa.display.specshow(magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始频谱图')
    
    # 应用阈值滤波和低频截止 (第一步预处理)
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[threshold_mask] = 0  # 阈值滤波
    filtered_magnitude[low_freq_mask, :] = 0  # 低频截止
    
    # 转换为dB用于显示
    filtered_magnitude_db = librosa.amplitude_to_db(filtered_magnitude, ref=np.max)
    
    # 阈值滤波后的频谱
    plt.subplot(2, 2, 2)
    librosa.display.specshow(filtered_magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'预处理1: 阈值滤波({threshold_db}dB)和低频截止(3kHz)')
    
    # 创建垂直方向（频率方向）的差分滤波器
    if order == 1:
        # 一阶差分滤波器
        y_diff_kernel = np.ones((height, 1)) / height
        y_diff_kernel[height//2:, :] *= -1  # 下半部分为负值
    else:
        # 二阶差分滤波器
        y_diff_kernel = np.zeros((height, 1))
        center = height // 2
        y_diff_kernel[center, 0] = 2 / height  # 中心为正
        if center > 0:
            y_diff_kernel[center-1, 0] = -1 / height  # 上边为负
        if center < height - 1:
            y_diff_kernel[center+1, 0] = -1 / height  # 下边为负
    
    # 应用垂直方向（频率轴）的差分滤波
    filtered_mag_y_diff = filtered_magnitude.copy()
    
    # 对每一个时间帧应用垂直方向的卷积
    for t in range(filtered_magnitude.shape[1]):
        # 获取当前时间帧的频率数据
        freq_column = filtered_magnitude[:, t:t+1]
        
        # 应用垂直方向的卷积
        if not np.all(freq_column == 0):  # 避免对全零列进行处理
            # 使用scipy.signal.convolve2d进行2D卷积
            filtered_column = signal.convolve2d(
                freq_column, y_diff_kernel, mode='same', boundary='symm'
            )
            
            # 混合原始信号和滤波后的信号
            filtered_mag_y_diff[:, t:t+1] = (
                freq_column * (1-intensity) + filtered_column * intensity
            )
    
    # 转换为二值图像用于膨胀操作
    binary_mask = filtered_mag_y_diff > 0
    
    # 创建频率自适应的膨胀操作
    # 随着频率升高增加膨胀程度
    # 获取频率轴的索引范围
    freq_indices = np.arange(binary_mask.shape[0])
    num_freq_bins = len(freq_indices)

    # 分成多个频率段，较高频率使用更大的膨胀程度
    # 使用3个频率段
    low_freq_idx = int(num_freq_bins * 0.25)    # 低频段 - 0-25%
    mid_freq_idx = int(num_freq_bins * 0.60)    # 中频段 - 25-60%
    # 高频段 - 60-100%

    # 创建不同大小的结构元素 - 恢复原来的参数
    low_struct = np.ones((2, 2))    # 低频使用小的结构元素
    mid_struct = np.ones((3, 3))    # 中频使用中等的结构元素
    high_struct = np.ones((4, 4))   # 高频使用大的结构元素

    # 分段应用膨胀操作
    dilated_mask = binary_mask.copy()

    # 低频段膨胀
    low_band = binary_mask[:low_freq_idx, :]
    if np.any(low_band):
        low_dilated = binary_dilation(low_band, structure=low_struct, iterations=1)
        dilated_mask[:low_freq_idx, :] = low_dilated

    # 中频段膨胀
    mid_band = binary_mask[low_freq_idx:mid_freq_idx, :]
    if np.any(mid_band):
        mid_dilated = binary_dilation(mid_band, structure=mid_struct, iterations=2)
        dilated_mask[low_freq_idx:mid_freq_idx, :] = mid_dilated

    # 高频段膨胀
    high_band = binary_mask[mid_freq_idx:, :]
    if np.any(high_band):
        high_dilated = binary_dilation(high_band, structure=high_struct, iterations=3)
        dilated_mask[mid_freq_idx:, :] = high_dilated

    print(f"应用频率自适应膨胀: 低频(2x2,迭代1次), 中频(3x3,迭代2次), 高频(4x4,迭代3次)")

    # 应用膨胀后的掩码
    dilated_magnitude = filtered_mag_y_diff.copy()
    dilated_magnitude[~dilated_mask] = 0
    
    # 显示膨胀后的频谱
    dilated_magnitude_db = librosa.amplitude_to_db(dilated_magnitude, ref=np.max)
    plt.subplot(2, 2, 3)
    librosa.display.specshow(dilated_magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('预处理2: 应用微分滤波+频率自适应膨胀操作')
    
    # 应用DBSCAN聚类
    # 先找到非零点的坐标
    nonzero_indices = np.where(dilated_magnitude > 0)
    if len(nonzero_indices[0]) > 0:  # 确保有非零点
        features = np.column_stack([nonzero_indices[0], nonzero_indices[1]])
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 应用DBSCAN聚类
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(features_scaled)
        
        # 找出样本点少于1000的聚类 (恢复原来的阈值)
        cluster_counts = np.bincount(clusters[clusters >= 0])
        small_clusters = np.where(cluster_counts < 1000)[0]
        
        print(f"DBSCAN聚类结果: 共{len(np.unique(clusters))-1}个聚类, 移除{len(small_clusters)}个小于1000点的聚类")
        
        # 创建一个掩码，标记要保留的点
        keep_mask = np.ones(len(clusters), dtype=bool)
        
        # 标记属于小聚类的点为需要移除
        for small_cluster in small_clusters:
            keep_mask[clusters == small_cluster] = False
        
        # 另外，标记噪声点（标签为-1）为需要移除
        keep_mask[clusters == -1] = False
        
        # 创建最终的频谱图
        final_magnitude = dilated_magnitude.copy()
        
        # 只保留要保留的点
        for i, (keep, (f, t)) in enumerate(zip(keep_mask, zip(nonzero_indices[0], nonzero_indices[1]))):
            if not keep:
                final_magnitude[f, t] = 0
    else:
        # 如果没有非零点，最终结果就是膨胀后的结果
        final_magnitude = dilated_magnitude.copy()
    
    # 显示DBSCAN聚类后的频谱
    final_magnitude_db = librosa.amplitude_to_db(final_magnitude, ref=np.max)
    plt.subplot(2, 2, 4)
    librosa.display.specshow(final_magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('最终结果: 应用DBSCAN聚类 (移除小于1000点的聚类)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "处理流程对比图.png"), dpi=300)
    plt.close()
    
    # 生成处理后的音频 - 直接重建
    final_D = final_magnitude * np.exp(1j * phase)
    final_audio = librosa.istft(final_D, hop_length=hop_length)
    
    # 保存处理后的音频
    filename = f"步骤4_聚类提取_h{height}_i{intensity:.1f}_阶{order}.wav"
    save_as_wav(final_audio, sr, filename, output_dir)
    
    # 对提取出的特征频谱进行积分处理（频率轴方向的累积和）
    # 由于微分滤波增强了频谱的变化特征，积分操作可以在某种程度上恢复频谱的原始形态
    integrated_magnitude = np.zeros_like(final_magnitude)

    # 对每个时间点的频率轴进行积分（累积和）
    for t in range(final_magnitude.shape[1]):
        freq_column = final_magnitude[:, t]
        # 从低频到高频的累积和
        integrated_magnitude[:, t] = np.cumsum(freq_column)

    # 归一化积分后的幅度
    if np.max(integrated_magnitude) > 0:
        integrated_magnitude = integrated_magnitude / np.max(integrated_magnitude)

    # 生成积分后的音频（作为参考）
    integrated_D = integrated_magnitude * np.exp(1j * phase)
    integrated_audio = librosa.istft(integrated_D, hop_length=hop_length)

    # 保存积分后的音频
    integrated_filename = f"步骤5_积分重建_h{height}_i{intensity:.1f}_阶{order}.wav"
    save_as_wav(integrated_audio, sr, integrated_filename, output_dir)

    # 创建负增益掩码 - 使用积分后的频谱作为掩码
    # 积分后的频谱中能量越大，应用的负增益越强
    norm_factor = np.max(integrated_magnitude) if np.max(integrated_magnitude) > 0 else 1.0
    gain_mask = integrated_magnitude / norm_factor
    
    # 设置负增益的最大值（例如-20dB）
    max_negative_gain_db = -20  # 可调整的参数
    
    # 将掩码转换为增益系数 (0到max_negative_gain_db的负增益)
    gain_factor = np.ones_like(gain_mask) - gain_mask * (1 - 10 ** (max_negative_gain_db/20))
    
    # 将增益应用到原始频谱
    negative_gain_D = original_D * gain_factor
    
    # 生成负增益处理后的音频
    negative_gain_audio = librosa.istft(negative_gain_D, hop_length=hop_length)
    
    # 保存负增益处理后的音频
    neg_gain_filename = f"最终处理_负增益{max_negative_gain_db}dB_h{height}_i{intensity:.1f}_阶{order}.wav"
    save_as_wav(negative_gain_audio, sr, neg_gain_filename, output_dir)
    
    # 创建负增益效果图
    plt.figure(figsize=(15, 12))
    
    # 原始频谱
    plt.subplot(3, 2, 1)
    librosa.display.specshow(magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始频谱图')
    
    # 提取的特征频谱（DBSCAN聚类后）
    plt.subplot(3, 2, 2)
    librosa.display.specshow(final_magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('提取的特征频谱 (DBSCAN聚类后)')
    
    # 积分后的频谱
    integrated_magnitude_db = librosa.amplitude_to_db(integrated_magnitude, ref=np.max)
    plt.subplot(3, 2, 3)
    librosa.display.specshow(integrated_magnitude_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('积分后的频谱 (用于负增益)')
    
    # 增益掩码
    plt.subplot(3, 2, 4)
    gain_mask_db = 20 * np.log10(gain_factor)
    gain_mask_db = np.clip(gain_mask_db, max_negative_gain_db, 0)
    librosa.display.specshow(gain_mask_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('增益掩码 (dB)')
    
    # 负增益处理后的频谱
    plt.subplot(3, 2, 5)
    negative_gain_magnitude = np.abs(negative_gain_D)
    negative_gain_db = librosa.amplitude_to_db(negative_gain_magnitude, ref=np.max)
    librosa.display.specshow(negative_gain_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'负增益处理后频谱 (最大{max_negative_gain_db}dB)')
    
    # 原始与处理后的幅度差异图
    plt.subplot(3, 2, 6)
    diff_magnitude = original_magnitude - negative_gain_magnitude
    diff_db = librosa.amplitude_to_db(np.abs(diff_magnitude), ref=np.max)
    librosa.display.specshow(diff_db, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始与处理后的差异')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "负增益处理效果图.png"), dpi=300)
    plt.close()
    
    # 保存处理过程中的中间音频
    # 阈值滤波后的音频
    filtered_D = filtered_magnitude * np.exp(1j * phase)
    filtered_audio = librosa.istft(filtered_D, hop_length=hop_length)
    save_as_wav(filtered_audio, sr, f"步骤1_阈值滤波_低频截止.wav", output_dir)
    
    # 微分滤波后的音频
    diff_D = filtered_mag_y_diff * np.exp(1j * phase)
    diff_audio = librosa.istft(diff_D, hop_length=hop_length)
    save_as_wav(diff_audio, sr, f"步骤2_微分滤波_h{height}_i{intensity:.1f}_阶{order}.wav", output_dir)
    
    # 膨胀后的音频
    dilated_D = dilated_magnitude * np.exp(1j * phase)
    dilated_audio = librosa.istft(dilated_D, hop_length=hop_length)
    save_as_wav(dilated_audio, sr, f"步骤3_膨胀处理.wav", output_dir)
    
    # 创建结果说明文件
    with open(os.path.join(output_dir, "处理结果说明.txt"), "w", encoding="utf-8") as f:
        f.write(f"频率轴微分滤波+频率自适应膨胀+负增益处理实验 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write("处理流程说明：\n")
        f.write("本实验使用下采样和以下处理流程：\n")
        f.write(f"0. 使用n_fft={n_fft}, hop_length={hop_length}的STFT参数减少运算量\n")
        f.write(f"1. 去除{threshold_db}dB以下的低幅度信号和3kHz以下的低频信号 (第一次预处理)\n")
        f.write(f"2. 应用垂直方向(频率轴)的微分滤波器，参数：高度={height}, 强度={intensity:.1f}, 阶数={order}\n")
        f.write(f"3. 应用频率自适应膨胀操作填充频谱中的缝隙:\n")
        f.write(f"   - 低频段(0-25%): 2x2结构元素, 迭代1次\n")
        f.write(f"   - 中频段(25-60%): 3x3结构元素, 迭代2次\n")
        f.write(f"   - 高频段(60-100%): 4x4结构元素, 迭代3次\n")
        f.write(f"4. 应用DBSCAN聚类算法，移除样本点少于1000的聚类和噪声点\n")
        f.write(f"5. 对提取的特征频谱进行积分处理（在频率轴方向累积），恢复频谱能量分布\n")
        f.write(f"6. 将积分后的频谱作为负增益掩码应用到原始音频上，频谱能量越大，负增益越强\n\n")
        
        f.write("参数详情:\n")
        f.write(f"- STFT参数: n_fft={n_fft}, hop_length={hop_length}\n")
        f.write(f"- 幅度阈值: {threshold_db}dB\n")
        f.write(f"- 低频截止: 3kHz\n")
        f.write(f"- 微分滤波器: 高度={height}, 强度={intensity:.1f}, 阶数={order}\n")
        f.write(f"- 膨胀操作: 频率自适应(低频2x2/中频3x3/高频4x4)\n")
        f.write(f"- DBSCAN参数: eps=0.5, min_samples=5, 小聚类阈值=1000\n")
        f.write(f"- 负增益处理: 最大负增益={max_negative_gain_db}dB，使用积分后频谱作为掩码\n\n")
        
        f.write("输出文件说明:\n")
        f.write(f"- 处理流程对比图.png: 显示处理的每个步骤的频谱图对比\n")
        f.write(f"- 负增益处理效果图.png: 显示负增益处理的过程和效果\n")
        f.write(f"- 步骤1_阈值滤波_低频截止.wav: 第一步预处理后的音频\n")
        f.write(f"- 步骤2_微分滤波_h{height}_i{intensity:.1f}_阶{order}.wav: 应用微分滤波后的音频\n")
        f.write(f"- 步骤3_膨胀处理.wav: 应用膨胀操作后的音频\n")
        f.write(f"- 步骤4_聚类提取_h{height}_i{intensity:.1f}_阶{order}.wav: 应用DBSCAN聚类后的特征音频\n")
        f.write(f"- 步骤5_积分重建_h{height}_i{intensity:.1f}_阶{order}.wav: 特征频谱积分后的音频\n")
        f.write(f"- 最终处理_负增益{max_negative_gain_db}dB_h{height}_i{intensity:.1f}_阶{order}.wav: 应用积分后的负增益处理的最终音频\n")
    
    return negative_gain_audio

def main():
    # 创建输出文件夹
    output_dir, hz_filter_dir, audio_dir = create_output_folders()
    
    # 指定使用的音频文件
    audio_file = "原始音频片段_01.mp3"
    audio_path = os.path.join(ROOT_DIR, "音频素材", audio_file)
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件 '{audio_path}' 不存在!")
        # 尝试在不同位置查找文件
        alt_path = os.path.join(ROOT_DIR, audio_file)
        if os.path.exists(alt_path):
            print(f"找到备用路径: {alt_path}")
            audio_path = alt_path
        else:
            # 搜索当前目录及子目录
            print("搜索音频文件...")
            for root, dirs, files in os.walk(ROOT_DIR):
                if audio_file in files:
                    audio_path = os.path.join(root, audio_file)
                    print(f"找到文件: {audio_path}")
                    break
            
            if not os.path.exists(audio_path):
                print("未找到指定音频文件，尝试查找任何MP3文件...")
                for root, dirs, files in os.walk(ROOT_DIR):
                    mp3_files = [f for f in files if f.endswith('.mp3')]
                    if mp3_files:
                        audio_file = mp3_files[0]
                        audio_path = os.path.join(root, audio_file)
                        print(f"使用找到的MP3文件: {audio_path}")
                        break
            
            if not os.path.exists(audio_path):
                print("无法找到任何MP3文件，请确保音频文件存在")
                return
    
    print(f"将使用音频文件: {audio_path}")
    
    # 使用下采样减少运算量，下采样到16kHz
    target_sr = 16000
    start_sec = 0
    end_sec = 30
    y, sr = load_audio(audio_path, start_sec=start_sec, end_sec=end_sec, target_sr=target_sr)
    
    # 保存原始音频的频谱图
    plot_spectrogram(y, sr, f'原始音频频谱图(下采样{target_sr}Hz): {audio_file} ({start_sec}-{end_sec}秒)', 
                    "原始频谱图.png", output_dir)
    
    # 应用整个处理流程
    threshold_db = -35
    processed_audio = apply_processing_pipeline(y, sr, audio_dir, threshold_db)
    
    print(f"\n频率轴微分滤波+频率自适应膨胀+负增益处理实验完成! 所有结果已保存到: {output_dir}")
    print(f"使用下采样率: {sr}Hz, 阈值: {threshold_db}dB")
    print(f"微分滤波参数: 高度=10, 强度=0.5, 阶数=2")
    print(f"完成膨胀操作、DBSCAN聚类和负增益处理")

if __name__ == "__main__":
    main()
