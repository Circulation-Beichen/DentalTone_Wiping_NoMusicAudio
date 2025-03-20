import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import librosa.display
import os
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 项目根目录路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def dynamic_deesser(audio, sr, threshold_db=-20, reduction_db=6, crossover=5000):
    """
    动态齿音消除器
    :param audio: 输入音频信号
    :param sr: 采样率
    :param threshold_db: 触发处理的阈值(dB)
    :param reduction_db: 增益衰减量
    :param crossover: 分频点(Hz)
    """
    # 使用Linkwitz-Riley滤波器分频[1](@ref)
    nyq = sr / 2
    b, a = signal.butter(4, crossover/nyq, 'high')
    high_freq = signal.filtfilt(b, a, audio)
    
    # 计算高频段包络
    env = librosa.amplitude_to_db(np.abs(librosa.stft(high_freq)), ref=np.max)
    mean_energy = np.mean(env[15:30])  # 聚焦6k-12kHz频段
    
    # 动态增益控制
    gain_reduction = np.ones_like(audio)
    if mean_energy > threshold_db:
        # 检测到齿音时创建衰减曲线
        window_size = int(sr * 0.02)  # 20ms窗口
        attenuation = 10**(-reduction_db/20)
        gain_reduction = np.convolve(
            np.ones(window_size)/window_size, 
            (high_freq > np.percentile(high_freq, 95)).astype(float),
            mode='same'
        ) * (1 - attenuation) + attenuation
    
    # 应用动态增益并混合低频
    processed_high = high_freq * gain_reduction
    b, a = signal.butter(4, crossover/nyq, 'low')
    low_freq = signal.filtfilt(b, a, audio)
    
    return low_freq + processed_high

def weighted_high_freq_average(audio, sr, freq_min=4000, freq_max=20000, peak_freq=6500):
    """
    计算音频的加权高频平均值，使用正态分布权重
    :param audio: 输入音频信号
    :param sr: 采样率
    :param freq_min: 考虑的最低频率(Hz)
    :param freq_max: 考虑的最高频率(Hz)
    :param peak_freq: 最高权重的频率中心(Hz)
    :return: 加权高频信息
    """
    # 计算STFT
    S = np.abs(librosa.stft(audio))
    
    # 转换为Mel频谱图
    mel_spec = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128, fmin=freq_min, fmax=freq_max)
    
    # 获取频率轴
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=freq_min, fmax=freq_max)
    
    # 创建基于正态分布的权重
    # 定义sigma使得在距离峰值2个sigma的地方权重降低到最大权重的一半
    sigma = (freq_max - freq_min) / 4  # 这样整个范围约为4个sigma
    min_weight = 1.0
    max_weight = 2.0 * min_weight  # 最高点是最低点的2倍
    
    weights = np.ones_like(mel_freqs) * min_weight
    
    for i, freq in enumerate(mel_freqs):
        # 计算正态分布权重（使用高斯函数）
        dist_squared = ((freq - peak_freq) / sigma) ** 2
        # 正态分布公式，但我们调整使得中心是max_weight，远处是min_weight
        weights[i] = min_weight + (max_weight - min_weight) * np.exp(-0.5 * dist_squared)
    
    # 应用权重
    weighted_mel = mel_spec * weights[:, np.newaxis]
    
    # 计算加权平均值
    weighted_avg = np.mean(weighted_mel, axis=0)
    
    return weighted_avg, mel_spec, weights

def plot_weighted_high_freq_analysis(weighted_avg, mel_spec, weights, sr, output_folder):
    """生成并保存高频加权分析图"""
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                           x_axis='time', y_axis='mel', sr=sr,
                           fmin=4000, fmax=20000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel频谱图 (4-20kHz)')
    
    plt.subplot(3, 1, 2)
    plt.plot(weights)
    plt.title('频率权重分布')
    plt.xlabel('Mel频率索引')
    plt.ylabel('权重')
    
    plt.subplot(3, 1, 3)
    plt.plot(weighted_avg)
    plt.title('加权高频平均值')
    plt.xlabel('时间帧')
    plt.ylabel('能量')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "高频加权分析.png"))
    plt.close()

def detect_high_freq_clusters_with_optics(audio, sr, time_start=1, time_end=3, freq_threshold=4000, max_reduction_db=10, min_samples=5, xi=0.05, min_cluster_size=0.05, output_dir=None):
    """
    使用OPTICS算法检测并处理指定时间范围内的高频聚类
    :param audio: 输入音频信号
    :param sr: 采样率
    :param time_start: 起始时间(秒)
    :param time_end: 结束时间(秒)
    :param freq_threshold: 频率阈值，只处理高于此频率的成分(Hz)
    :param max_reduction_db: 最大负增益值(dB)，负值
    :param min_samples: OPTICS参数，用于确定核心点的最小样本数
    :param xi: OPTICS参数，用于提取分层聚类的陡度阈值
    :param min_cluster_size: OPTICS参数，最小聚类大小(占总样本比例)
    :param output_dir: 输出文件夹路径
    :return: 处理后的音频
    """
    # 创建输出文件夹
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "处理结果")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建OPTICS子文件夹
    optics_dir = os.path.join(output_dir, "OPTICS分析")
    if not os.path.exists(optics_dir):
        os.makedirs(optics_dir)
    
    # 将时间转换为样本索引
    start_sample = int(time_start * sr)
    end_sample = int(time_end * sr)
    
    # 提取目标时间范围内的音频片段
    target_segment = audio[start_sample:end_sample]
    
    # 计算加权高频平均值
    weighted_avg, mel_spec, weights = weighted_high_freq_average(
        target_segment, sr, freq_min=4000, freq_max=20000, peak_freq=6500
    )
    
    # 绘制加权高频平均值图
    plot_weighted_high_freq_analysis(weighted_avg, mel_spec, weights, sr, optics_dir)
    
    # 计算短时傅里叶变换
    D = librosa.stft(target_segment)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 获取频率和时间轴
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
    
    # 只考虑高于频率阈值的部分的频谱平均值
    high_freq_indices = np.where(freqs >= freq_threshold)[0]
    high_freq_magnitudes = magnitude[high_freq_indices, :]
    
    # 对每个时间帧计算高频平均幅度
    time_features = []
    for t in range(high_freq_magnitudes.shape[1]):
        frame_magnitude = high_freq_magnitudes[:, t]
        # 计算当前时间帧的高频平均幅度
        avg_magnitude = np.mean(frame_magnitude)
        # 将时间和平均幅度作为特征
        time_features.append([times[t], avg_magnitude, weighted_avg[min(t, len(weighted_avg)-1)]])
    
    # 转换为numpy数组
    time_features_array = np.array(time_features)
    
    if len(time_features_array) == 0:
        print("未检测到有效的时间帧数据")
        return audio
    
    # 归一化特征，只使用幅度和加权值（排除时间）
    features_for_clustering = time_features_array[:, 1:]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)
    
    # 使用OPTICS算法进行聚类
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics.fit(features_scaled)
    
    # 获取聚类标签和可达距离
    labels = optics.labels_
    reachability = optics.reachability_
    ordering = optics.ordering_
    
    # 可达距离可视化
    plt.figure(figsize=(12, 6))
    plt.plot(reachability[ordering])
    plt.title('OPTICS算法可达距离图')
    plt.xlabel('样本点（按聚类顺序）')
    plt.ylabel('可达距离')
    
    # 在图上标记聚类
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:  # 噪声点
            col = [0, 0, 0, 1]  # 黑色
            
        class_mask = labels[ordering] == k
        plt.plot(np.where(class_mask)[0], reachability[ordering][class_mask], 'o', 
                 markerfacecolor=col, markeredgecolor='k', markersize=5)
        
    plt.savefig(os.path.join(optics_dir, "OPTICS_可达距离图.png"))
    plt.close()
    
    # 在时间轴上可视化聚类结果
    plt.figure(figsize=(12, 6))
    for i, (t, mag, _) in enumerate(time_features_array):
        if labels[i] == -1:
            color = [0, 0, 0, 1]  # 噪声点为黑色
        else:
            color = plt.cm.viridis(labels[i] / max(1, max(labels)))
            
        plt.scatter(t, mag, c=[color], s=20, alpha=0.8)
    
    plt.title('时间轴上的高频聚类')
    plt.xlabel('时间 (秒)')
    plt.ylabel('高频能量')
    plt.savefig(os.path.join(optics_dir, "高频聚类时间分布.png"))
    plt.close()
    
    # 找出各个聚类的中心
    cluster_centers = []
    for cluster_id in unique_labels:
        if cluster_id != -1:  # 跳过噪声点
            # 找出该聚类中的所有点
            cluster_points = np.where(labels == cluster_id)[0]
            
            if len(cluster_points) > 0:
                # 计算聚类中心的时间位置（使用能量加权平均）
                energy_weights = features_for_clustering[cluster_points, 0] * features_for_clustering[cluster_points, 1]
                weighted_sum = np.sum(time_features_array[cluster_points, 0] * energy_weights)
                weight_sum = np.sum(energy_weights)
                cluster_time = weighted_sum / weight_sum if weight_sum > 0 else np.mean(time_features_array[cluster_points, 0])
                
                # 找到最接近计算出的时间的时间帧索引
                t_idx = np.argmin(np.abs(times - cluster_time))
                
                # 计算该聚类的平均能量
                avg_energy = np.mean(features_for_clustering[cluster_points, 0])
                
                cluster_centers.append((t_idx, avg_energy, cluster_id, cluster_time))
    
    print(f"检测到的{len(cluster_centers)}个高频时间聚类中心:")
    for i, (t_idx, avg_energy, cluster_id, t_time) in enumerate(cluster_centers):
        print(f"聚类 #{cluster_id}: 时间={t_time:.2f}秒, 平均高频能量={librosa.amplitude_to_db(avg_energy):.1f}dB")
    
    # 创建处理后的STFT矩阵
    processed_magnitude = magnitude.copy()
    
    # 获取mel频率映射到STFT频率的索引
    stft_freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-2)
    
    # 创建一个与stft_freqs相同大小的权重数组
    stft_weights = np.ones_like(stft_freqs)
    
    # 为高频部分(>=4kHz)计算与之前相同的正态分布权重
    high_freq_stft_indices = np.where(stft_freqs >= freq_threshold)[0]
    peak_freq = 6500  # 与之前的加权分析相同
    sigma = (20000 - 4000) / 4  # 与加权分析相同
    min_weight = 1.0
    max_weight = 2.0 * min_weight  # 最高点是最低点的2倍
    
    for i in high_freq_stft_indices:
        freq = stft_freqs[i]
        # 计算正态分布权重
        dist_squared = ((freq - peak_freq) / sigma) ** 2
        stft_weights[i] = min_weight + (max_weight - min_weight) * np.exp(-0.5 * dist_squared)
    
    # 标准化权重，使最大值为1
    stft_weights = stft_weights / np.max(stft_weights)
    
    # 在每个聚类中心时间的高频范围上应用衰减
    for t_idx, _, _, _ in cluster_centers:
        # 定义时间窗口范围
        t_window = 8  # 时间范围窗口大小
        
        # 安全地获取周围区域
        t_start = max(0, t_idx - t_window)
        t_end = min(processed_magnitude.shape[1], t_idx + t_window + 1)
        
        # 只对高频部分应用增益，和使用与识别时相同的权重分布
        for f_idx in high_freq_indices:
            # 获取该频率的权重，用于调整增益
            freq_weight = stft_weights[f_idx]
            
            # 计算该频率点的负增益，权重越高，负增益越大，但不超过最大负增益
            reduction_db = max_reduction_db * freq_weight
            gain_factor = 10 ** (-reduction_db / 20)  # 将dB转换为线性增益因子
            
            # 创建高斯时间衰减窗口，使得衰减在中心时间最强，向外逐渐减弱
            t_coords = np.arange(t_start, t_end)
            gaussian_window = np.exp(-0.5 * (((t_coords - t_idx) / (t_window / 2)) ** 2))
            
            # 应用衰减，中心时间点衰减最大，周围逐渐减弱
            attenuation_factor = 1 - ((1 - gain_factor) * gaussian_window)
            processed_magnitude[f_idx, t_start:t_end] *= attenuation_factor
    
    # 重建音频信号
    processed_D = processed_magnitude * np.exp(1j * phase)
    processed_segment = librosa.istft(processed_D)
    
    # 将处理后的片段替换回原始音频
    result = audio.copy()
    result[start_sample:start_sample + len(processed_segment)] = processed_segment
    
    return result
