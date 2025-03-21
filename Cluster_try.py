import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import time
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib
from matplotlib.font_manager import FontProperties
from scipy import ndimage
import soundfile as sf
from sklearn.metrics import pairwise_distances
from scipy import signal
warnings.filterwarnings('ignore')

# 移除GPU相关导入
HAS_GPU = False
print("使用CPU运行OPTICS聚类算法")

# 设置matplotlib使用中文字体，解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 设置输出目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "OPTICS聚类结果")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 生成唯一的时间戳用于文件名，防止覆盖
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

def load_audio(audio_path):
    """加载音频文件"""
    print(f"正在加载音频: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"音频加载成功! 采样率: {sr}Hz, 长度: {len(y)/sr:.2f}秒")
    return y, sr

def preprocess_audio(audio, sr, threshold_db=-20, width=7, intensity=0.7, order=1):
    """
    使用指定参数预处理音频
    
    Args:
        audio: 音频数据
        sr: 采样率
        threshold_db: 幅度阈值（dB）
        width: 滤波器宽度
        intensity: 滤波器强度
        order: 微分阶数
    
    Returns:
        filtered_mag: 滤波后的幅度
        phase: 相位信息
        non_zero_mask: 不为零部分的掩码
        magnitude_db: 原始幅度（dB）
        filtered_magnitude_db: 滤波后的幅度（dB）
        freqs: 频率轴
        times: 时间轴
    """
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 获取STFT频率轴和时间轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-2)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr)
    
    # 步骤1: 删除3kHz以下的频率
    low_freq_mask = freqs < 3000
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[low_freq_mask, :] = 0
    
    # 步骤2: 删除低于阈值的幅度
    magnitude_db = librosa.amplitude_to_db(filtered_magnitude, ref=np.max)
    low_amp_mask = magnitude_db < threshold_db
    filtered_magnitude[low_amp_mask] = 0
    
    # 步骤3: 应用微分滤波器
    # 创建差分滤波器
    if order == 1:
        # 一阶微分滤波器
        diff_kernel = np.ones(width) / width
        diff_kernel[width//2:] *= -1  # 将后半部分设为负值，创建差分效果
    else:
        # 二阶微分滤波器
        diff_kernel = np.zeros(width)
        if width >= 3:
            center = width // 2
            # 二阶差分: 中心为2，两边为-1
            diff_kernel[center] = 2 / width
            if center > 0:
                diff_kernel[center-1] = -1 / width
            if center < width - 1:
                diff_kernel[center+1] = -1 / width
    
    # 应用差分滤波
    filtered_mag = filtered_magnitude.copy()
    for f_idx in range(filtered_mag.shape[0]):
        if not (low_freq_mask[f_idx] or np.all(filtered_mag[f_idx, :] == 0)):
            # 应用卷积
            filtered_row = np.convolve(filtered_mag[f_idx, :], diff_kernel, mode='same')
            # 应用强度参数
            filtered_mag[f_idx, :] = filtered_mag[f_idx, :] * (1-intensity) + filtered_row * intensity
    
    # 转换为dB尺度
    filtered_magnitude_db = librosa.amplitude_to_db(filtered_mag, ref=np.max)
    
    # 获取不为零的掩码
    non_zero_mask = filtered_mag > 0
    
    return filtered_mag, phase, non_zero_mask, magnitude_db, filtered_magnitude_db, freqs, times

def extract_features_from_magnitude(filtered_mag, freqs, times):
    """
    从幅度矩阵中提取特征用于聚类
    
    Args:
        filtered_mag: 滤波后的幅度
        freqs: 频率轴
        times: 时间轴
    
    Returns:
        features: 用于聚类的特征，格式为[[时间, 频率, 幅度], ...]
    """
    # 转换为dB尺度
    filtered_magnitude_db = librosa.amplitude_to_db(filtered_mag, ref=np.max)
    
    # 获取非零幅度的掩码
    non_zero_mask = filtered_mag > 0
    
    # 提取特征用于聚类
    features = []
    for f_idx in range(filtered_mag.shape[0]):
        for t_idx in range(filtered_mag.shape[1]):
            if non_zero_mask[f_idx, t_idx]:
                # 保存时间、频率和幅度
                features.append([times[t_idx], freqs[f_idx], filtered_magnitude_db[f_idx, t_idx]])
    
    # 转换为numpy数组
    features = np.array(features)
    
    return features

def chessboard_metric(x, y, time_scale, mel_scale):
    """
    基于梅尔尺度的棋盘距离（Chebyshev距离）
    
    Args:
        x, y: 两个点的坐标，格式为[时间, 频率]
        time_scale: 时间轴的缩放因子
        mel_scale: 梅尔尺度的缩放因子
    
    Returns:
        梅尔尺度下的棋盘距离
    """
    # 将频率转换为梅尔尺度
    x_mel = librosa.hz_to_mel(x[1]) * mel_scale
    y_mel = librosa.hz_to_mel(y[1]) * mel_scale
    
    # 缩放时间坐标
    x_time = x[0] * time_scale
    y_time = y[0] * time_scale
    
    # 计算棋盘距离（最大绝对差值）
    return max(abs(x_time - y_time), abs(x_mel - y_mel))

def perform_optics_clustering(features, time_scale, mel_scale, max_eps=None, min_samples=10):
    """
    使用OPTICS算法对预处理后的数据进行聚类，基于梅尔尺度的距离
    
    Args:
        features: 用于聚类的特征，格式为[[时间, 频率, 幅度], ...]
        time_scale: 时间轴的缩放因子
        mel_scale: 梅尔尺度的缩放因子
        max_eps: 最大可达距离，如果为None则使用默认值
        min_samples: OPTICS算法的min_samples参数
    
    Returns:
        labels: 聚类标签
    """
    # 如果没有足够的数据点，则跳过聚类
    if len(features) < min_samples:
        print(f"没有足够的数据点用于聚类")
        return np.full(len(features), -1)  # 所有点都标记为噪声
    
    # 提取用于聚类的时间和频率特征
    X = features[:, :2]
    
    # 计算最小时间间隔和频率间隔
    times = X[:, 0]
    freqs = X[:, 1]
    
    # 计算时间和频率的最小间隔（如果有多个点）
    if len(times) > 1:
        # 对时间和频率进行排序
        sorted_times = np.sort(times)
        sorted_freqs = np.sort(freqs)
        
        # 计算相邻点的差值
        time_diffs = np.diff(sorted_times)
        freq_diffs = np.diff(sorted_freqs)
        
        # 找出非零的最小间隔
        non_zero_time_diffs = time_diffs[time_diffs > 0]
        non_zero_freq_diffs = freq_diffs[freq_diffs > 0]
        
        # 计算最小间隔（如果存在非零间隔）
        min_time_interval = np.min(non_zero_time_diffs) if len(non_zero_time_diffs) > 0 else 0.01
        min_freq_interval = np.min(non_zero_freq_diffs) if len(non_zero_freq_diffs) > 0 else 1.0
        
        print(f"最小时间间隔: {min_time_interval:.6f} 秒")
        print(f"最小频率间隔: {min_freq_interval:.2f} Hz")
    else:
        # 默认值
        min_time_interval = 0.01  # 10毫秒
        min_freq_interval = 1.0   # 1赫兹
    
    # 设置最大可达距离
    if max_eps is None:
        # 基于最小间隔的最大距离：
        # 时间轴：50个最小时间间隔
        # 频率轴：200个mel单位
        max_time_dist = 50 * min_time_interval  # 50个最小时间间隔
        max_mel_dist = 200.0  # 梅尔单位
        
        # 将两个轴的最大距离结合（取较小值作为截断距离）
        max_eps = min(max_time_dist * time_scale, max_mel_dist * mel_scale)
    
    print(f"使用最大可达距离: {max_eps:.6f}")
    print(f"数据点总数: {len(X)}")
    
    # 预计算所有点的梅尔频率值，避免重复计算
    mel_freqs = np.array([librosa.hz_to_mel(x[1]) for x in X])
    
    # 计算OPTICS聚类所需的距离矩阵
    print("计算距离矩阵...")
    n_samples = len(X)
    
    # 使用有限的大值代替无穷大，避免OPTICS算法出错
    MAX_DISTANCE = 1e6  # 一个足够大但有限的值
    distance_matrix = np.full((n_samples, n_samples), MAX_DISTANCE)
    
    # 对角线元素设为0
    np.fill_diagonal(distance_matrix, 0)
    
    # 使用窗函数约束，只计算在时间和频率上都接近的点对之间的距离
    # 窗口大小基于最小间隔
    time_window = max_time_dist  # 50个最小时间间隔
    mel_window = max_mel_dist    # 200个mel单位
    
    print(f"时间窗口: {time_window:.6f} 秒")
    print(f"梅尔窗口: {mel_window:.2f} mel单位")
    
    # 统计有效计算的距离对数
    valid_distances = 0
    
    # 使用numpy向量化操作加速计算
    start_time = time.time()
    
    for i in range(n_samples):
        time_i = X[i, 0]
        mel_i = mel_freqs[i]
        
        for j in range(i+1, n_samples):  # 只计算上三角矩阵
            time_j = X[j, 0]
            mel_j = mel_freqs[j]
            
            # 检查是否在窗口内
            time_diff = abs(time_i - time_j)
            mel_diff = abs(mel_i - mel_j)
            
            if time_diff <= time_window and mel_diff <= mel_window:
                # 在窗口内，计算棋盘距离
                scaled_time_diff = time_diff * time_scale
                scaled_mel_diff = mel_diff * mel_scale
                
                # 计算棋盘距离（最大值）
                dist = max(scaled_time_diff, scaled_mel_diff)
                
                # 如果距离小于最大可达距离，则保存
                if dist <= max_eps:
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # 对称矩阵
                    valid_distances += 1
    
    end_time = time.time()
    print(f"距离计算耗时: {end_time - start_time:.2f}秒")
    print(f"有效距离对数: {valid_distances} (总共可能的对数: {n_samples * (n_samples - 1) // 2})")
    
    # 检查距离矩阵是否有效（是否所有点至少有一个邻居）
    isolated_points = np.sum(np.all(distance_matrix >= MAX_DISTANCE, axis=1))
    if isolated_points > 0:
        print(f"警告: 有{isolated_points}个孤立点没有邻居")
    
    # 如果所有点都是孤立的，无法进行聚类
    if isolated_points == n_samples:
        print("错误: 所有点都是孤立的，无法进行聚类")
        return np.full(n_samples, -1)
    
    # 使用OPTICS进行聚类，设置较小的最大可达距离以减少计算量
    optics = OPTICS(
        min_samples=min_samples, 
        metric='precomputed',
        max_eps=max_eps,
        cluster_method='xi',
        xi=0.05
    )
    
    # 执行聚类
    print("执行OPTICS聚类...")
    start_time = time.time()
    
    try:
        optics.fit(distance_matrix)
        # 获取聚类标签
        labels = optics.labels_
        
        end_time = time.time()
        print(f"CPU聚类耗时: {end_time - start_time:.2f}秒")
        
        # 统计聚类结果
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"聚类结果: {n_clusters}个类别，噪声点: {np.sum(labels == -1)}/{n_samples}")
        
        return labels
    except Exception as e:
        print(f"聚类过程出错: {e}")
        import traceback
        traceback.print_exc()
        # 尝试处理错误并返回默认标签
        return np.full(n_samples, -1)

def filter_clusters_by_frequency_span(features, labels):
    """
    过滤聚类，保留频率跨度不小于最大跨度二分之一的聚类
    
    Args:
        features: 用于聚类的特征，格式为[[时间, 频率, 幅度], ...]
        labels: 聚类标签
    
    Returns:
        filtered_labels: 过滤后的聚类标签
        cluster_centers: 聚类中心，格式为[[时间, 频率], ...]
    """
    # 复制标签以保存原始聚类
    filtered_labels = labels.copy()
    
    # 计算每个聚类的频率跨度
    unique_labels = np.unique(labels)
    max_freq_span = 0
    cluster_freq_spans = {}
    
    for label in unique_labels:
        if label != -1:  # 跳过噪声点
            # 获取该聚类中的所有点
            cluster_points = features[labels == label]
            
            # 计算频率跨度
            freq_span = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            cluster_freq_spans[label] = freq_span
            
            # 更新最大频率跨度
            max_freq_span = max(max_freq_span, freq_span)
    
    # 过滤聚类，保留跨度不小于最大跨度二分之一的聚类
    threshold_span = max_freq_span / 2
    print(f"最大频率跨度: {max_freq_span:.2f} Hz, 阈值: {threshold_span:.2f} Hz")
    
    cluster_centers = []
    
    for label in unique_labels:
        if label != -1:  # 跳过噪声点
            # 获取该聚类中的所有点
            cluster_points = features[labels == label]
            freq_span = cluster_freq_spans[label]
            
            if freq_span >= threshold_span:
                # 计算聚类中心（加权平均）
                weights = 10 ** (cluster_points[:, 2] / 20)  # 将dB转换为线性幅度作为权重
                weighted_sum = np.sum(cluster_points[:, :2] * weights[:, np.newaxis], axis=0)
                weight_sum = np.sum(weights)
                
                if weight_sum > 0:
                    center = weighted_sum / weight_sum
                    cluster_centers.append(center)
            else:
                # 将小跨度聚类的点标记为噪声
                filtered_labels[labels == label] = -1
    
    return filtered_labels, cluster_centers

def filter_clusters_by_min_frequency_span(features, labels, min_span=6000):
    """
    过滤聚类，保留频率跨度大于指定值的聚类
    
    Args:
        features: 用于聚类的特征，格式为[[时间, 频率, 幅度], ...]
        labels: 聚类标签
        min_span: 最小频率跨度（Hz）
    
    Returns:
        filtered_labels: 过滤后的聚类标签
    """
    # 复制标签以保存原始聚类
    filtered_labels = labels.copy()
    
    # 计算每个聚类的频率跨度
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label != -1:  # 跳过噪声点
            # 获取该聚类中的所有点
            cluster_points = features[labels == label]
            
            # 计算频率跨度
            freq_span = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            
            # 如果跨度小于最小要求，将点标记为噪声
            if freq_span < min_span:
                filtered_labels[labels == label] = -1
    
    return filtered_labels

def apply_gain_reduction(audio, sr, features, labels, phase, max_reduction_db=20):
    """
    根据聚类结果对原音频应用负增益
    
    Args:
        audio: 原始音频数据
        sr: 采样率
        features: 用于聚类的特征，格式为[[时间, 频率, 幅度], ...]
        labels: 聚类标签
        phase: 相位信息
        max_reduction_db: 最大负增益值（dB）
    
    Returns:
        processed_audio: 处理后的音频
        gain_factors: 应用的增益因子
    """
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    
    # 创建增益矩阵（初始化为1，表示不改变）
    gain_factors = np.ones_like(magnitude)
    
    # 获取时间和频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-2)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr)
    
    # 找出有效的聚类（非噪声点）
    valid_cluster_points = features[labels != -1]
    
    if len(valid_cluster_points) > 0:
        # 找出有效聚类点中的最大幅度值
        max_amplitude_db = np.max(valid_cluster_points[:, 2])
        min_amplitude_db = np.min(valid_cluster_points[:, 2])
        amplitude_range = max_amplitude_db - min_amplitude_db
        
        # 对于每个有效聚类点，计算负增益
        for point in valid_cluster_points:
            time, freq, amplitude_db = point
            
            # 将幅度标准化到[0,1]范围
            normalized_amplitude = (amplitude_db - min_amplitude_db) / amplitude_range if amplitude_range > 0 else 0
            
            # 根据幅度计算负增益，幅度越大，负增益越大
            reduction_db = max_reduction_db * normalized_amplitude
            
            # 将dB转换为线性增益因子
            gain_factor = 10 ** (-reduction_db / 20)
            
            # 找到最近的时间和频率索引
            t_idx = np.argmin(np.abs(times - time))
            f_idx = np.argmin(np.abs(freqs - freq))
            
            # 应用增益因子
            if 0 <= f_idx < magnitude.shape[0] and 0 <= t_idx < magnitude.shape[1]:
                gain_factors[f_idx, t_idx] = gain_factor
    
    # 应用增益
    processed_magnitude = magnitude * gain_factors
    
    # 重建音频
    processed_D = processed_magnitude * np.exp(1j * phase)
    processed_audio = librosa.istft(processed_D)
    
    return processed_audio, gain_factors

def visualize_clustering_results(features, labels, clusters_name, output_path):
    """
    可视化聚类结果
    
    Args:
        features: 用于聚类的特征，格式为[[时间, 频率, 幅度], ...]
        labels: 聚类标签
        clusters_name: 聚类名称，用于标题
        output_path: 输出文件路径
    """
    if len(features) == 0:
        print(f"没有数据可供可视化")
        return
    
    # 计算总共有多少个聚类
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"检测到{n_clusters}个聚类")
    
    # 创建一个新的图形
    plt.figure(figsize=(12, 8))
    
    # 使用鲜艳的颜色映射来绘制聚类
    cmap = plt.cm.tab10 if n_clusters <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(10, n_clusters)))
    
    # 如果有噪声点，则先绘制噪声点（灰色）
    noise_points = features[labels == -1]
    if len(noise_points) > 0:
        plt.scatter(
            noise_points[:, 0], 
            noise_points[:, 1], 
            c='lightgray', 
            marker='.', 
            s=30, 
            alpha=0.5,
            label='噪声'
        )
    
    # 为每个聚类绘制数据点
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue  # 噪声点已经绘制
            
        # 获取该聚类中的所有点
        cluster_points = features[labels == label]
        
        # 绘制数据点，颜色根据聚类标签确定
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            c=[colors[i % len(colors)]], 
            marker='o', 
            s=50, 
            alpha=0.8,
            label=f'聚类 {label}'
        )
        
        # 计算并绘制聚类中心
        center = np.mean(cluster_points[:, :2], axis=0)
        plt.scatter(
            center[0], 
            center[1], 
            c=[colors[i % len(colors)]], 
            marker='x', 
            s=200, 
            linewidths=3,
            edgecolors='black'
        )
    
    # 设置图表属性
    plt.title(f'{clusters_name}聚类结果', fontsize=16)
    plt.xlabel('时间 (秒)', fontsize=14)
    plt.ylabel('频率 (Hz)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 使用对数刻度显示频率轴
    plt.yscale('log')
    
    # 添加图例
    if n_clusters > 0:
        plt.legend(loc='upper right', fontsize=12)
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"已保存聚类结果到 {output_path}")

def main():
    # 指定音频文件
    audio_file = "原始音频片段_01.mp3"
    audio_path = os.path.join(ROOT_DIR, audio_file)
    
    # 创建输出文件夹
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载音频
    audio, sr = load_audio(audio_path)
    
    # 第一步：使用参数 (-20dB阈值，w=7，i=0.7，阶=1) 进行预处理
    print("第一次预处理：使用参数 (-20dB阈值，w=7，i=0.7，阶=1)")
    filtered_mag1, phase1, non_zero_mask1, magnitude_db1, filtered_magnitude_db1, freqs1, times1 = preprocess_audio(
        audio, sr, threshold_db=-20, width=7, intensity=0.7, order=1
    )
    
    # 从预处理结果中提取特征
    features1 = extract_features_from_magnitude(filtered_mag1, freqs1, times1)
    
    # 计算时间和频率的梅尔尺度因子
    time_scale = 1.0  # 不缩放时间轴
    mel_scale = 1.0  # 不缩放梅尔轴
    
    print(f"时间缩放因子: {time_scale}, 梅尔缩放因子: {mel_scale}")
    
    # 使用CPU进行OPTICS聚类
    print("第一次聚类：使用梅尔尺度的棋盘距离进行OPTICS聚类")
    labels1 = perform_optics_clustering(features1, time_scale, mel_scale, min_samples=5)
    
    # 过滤聚类，保留频率跨度足够大的聚类
    print("过滤第一次聚类，保留频率跨度不小于最大跨度二分之一的聚类")
    filtered_labels1, cluster_centers1 = filter_clusters_by_frequency_span(features1, labels1)
    
    # 可视化第一次聚类结果
    print("可视化第一次聚类结果")
    visualize_clustering_results(
        features1, 
        filtered_labels1, 
        "第一次", 
        os.path.join(output_dir, f"第一次聚类结果_{TIMESTAMP}.png")
    )
    
    # 第二步：使用参数 (-30dB阈值，w=7，i=0.5，阶=2) 进行第二次预处理
    print("第二次预处理：使用参数 (-30dB阈值，w=7，i=0.5，阶=2)")
    filtered_mag2, phase2, non_zero_mask2, magnitude_db2, filtered_magnitude_db2, freqs2, times2 = preprocess_audio(
        audio, sr, threshold_db=-30, width=7, intensity=0.5, order=2
    )
    
    # 从预处理结果中提取特征
    features2 = extract_features_from_magnitude(filtered_mag2, freqs2, times2)
    
    # 使用CPU进行OPTICS聚类
    print("第二次聚类：使用梅尔尺度的棋盘距离进行OPTICS聚类")
    labels2 = perform_optics_clustering(features2, time_scale, mel_scale, min_samples=5)
    
    # 过滤聚类，仅保留频率跨度大于6kHz的聚类
    print("过滤第二次聚类，仅保留频率跨度大于6kHz的聚类")
    filtered_labels2 = filter_clusters_by_min_frequency_span(features2, labels2, min_span=6000)
    
    # 可视化第二次聚类结果
    print("可视化第二次聚类结果")
    visualize_clustering_results(
        features2, 
        filtered_labels2, 
        "第二次", 
        os.path.join(output_dir, f"第二次聚类结果_{TIMESTAMP}.png")
    )
    
    # 对原音频应用负增益
    print("对原音频应用基于聚类的负增益")
    processed_audio, gain_factors = apply_gain_reduction(
        audio, sr, features2, filtered_labels2, phase2, max_reduction_db=20
    )
    
    # 保存处理后的音频
    output_audio_path = os.path.join(output_dir, f"处理后音频_{TIMESTAMP}.wav")
    sf.write(output_audio_path, processed_audio, sr)
    print(f"处理后音频已保存至: {output_audio_path}")
    
    # 创建处理说明
    with open(os.path.join(output_dir, f"处理说明_{TIMESTAMP}.txt"), 'w', encoding='utf-8') as f:
        f.write("OPTICS聚类处理说明\n")
        f.write("="*30 + "\n\n")
        f.write(f"处理参数:\n")
        f.write(f"- 音频文件: {audio_file}\n")
        f.write(f"- 第一次预处理: 阈值=-20dB, 滤波器宽度=7, 强度=0.7, 阶数=1\n")
        f.write(f"- 第二次预处理: 阈值=-30dB, 滤波器宽度=7, 强度=0.5, 阶数=2\n")
        f.write(f"- 聚类距离: 梅尔尺度下的棋盘距离\n")
        f.write(f"- 时间窗口: 50个最小时间间隔\n")
        f.write(f"- 梅尔窗口: 200个mel单位\n")
        f.write(f"- 聚类过滤: 频率跨度>6kHz\n")
        f.write(f"- 最大负增益: 20dB\n")
        f.write(f"- 计算设备: CPU\n\n")
        f.write("处理步骤:\n")
        f.write("1. 使用(-20dB阈值，w=7，i=0.7，阶=1)预处理音频\n")
        f.write("2. 使用OPTICS算法对预处理结果进行聚类，使用梅尔尺度的棋盘距离\n")
        f.write("3. 保留频率跨度不小于最大跨度二分之一的聚类\n")
        f.write("4. 使用(-30dB阈值，w=7，i=0.5，阶=2)再次预处理音频\n")
        f.write("5. 使用OPTICS算法对第二次预处理结果进行聚类\n")
        f.write("6. 仅保留频率跨度大于6kHz的聚类\n")
        f.write("7. 对原音频应用负增益，聚类中幅度越大的点负增益越大（最大-20dB）\n\n")
        f.write("计算优化:\n")
        f.write("1. 使用梅尔尺度的距离计算，更符合人耳感知\n")
        f.write("2. 使用窗函数限制距离计算，只考虑时间和频率上接近的点\n")
        f.write("3. 基于最小间隔单位设置窗口大小，更适应数据特性\n\n")
        f.write("处理结果:\n")
        f.write(f"- 第一次聚类结果: {os.path.join(output_dir, f'第一次聚类结果_{TIMESTAMP}.png')}\n")
        f.write(f"- 第二次聚类结果: {os.path.join(output_dir, f'第二次聚类结果_{TIMESTAMP}.png')}\n")
        f.write(f"- 处理后音频: {output_audio_path}\n")
    
    print(f"处理完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    main() 