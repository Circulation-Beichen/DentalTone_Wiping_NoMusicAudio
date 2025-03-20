import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import time
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import matplotlib
from matplotlib.font_manager import FontProperties
from scipy import stats
from pydub import AudioSegment
import io
from scipy.ndimage import minimum_filter
warnings.filterwarnings('ignore')

# 设置matplotlib使用微软雅黑或其他中文字体，解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 设置输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OPTICS聚类结果")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 生成唯一的时间戳用于文件名，防止覆盖
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

def normalize_distances(distances):
    """
    对距离进行归一化处理，保持数据的相对关系
    
    参数:
    - distances: 需要归一化的距离数组
    
    返回:
    - 归一化后的距离数组
    """
    # 处理无穷大值
    valid_mask = np.isfinite(distances)
    valid_distances = distances[valid_mask]
    
    if len(valid_distances) == 0:
        return distances  # 如果全都是无穷大，则直接返回
    
    # 使用稳健的统计方法进行归一化
    # 1. 计算四分位数范围（IQR）来检测异常值
    q1 = np.percentile(valid_distances, 25)
    q3 = np.percentile(valid_distances, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 2. 将超出范围的值截断到边界值
    normalized = np.copy(distances)
    normalized[valid_mask] = np.clip(valid_distances, lower_bound, upper_bound)
    
    # 3. 使用MinMaxScaler进行归一化，但保持相对关系
    scaler = MinMaxScaler()
    normalized[valid_mask] = scaler.fit_transform(normalized[valid_mask].reshape(-1, 1)).flatten()
    
    # 4. 对结果进行轻微的指数变换，以增强对比度
    normalized[valid_mask] = np.power(normalized[valid_mask], 0.7)
    
    # 5. 保持无穷大的值不变
    normalized[~valid_mask] = distances[~valid_mask]
    
    return normalized

def mel_spectrogram_with_optics_clustering(audio_path, window_size=10, window_step=5, freq_threshold=3500):
    """
    使用OPTICS对音频中频率高于阈值的部分进行聚类，并可视化结果
    
    参数:
    - audio_path: 音频文件路径
    - window_size: 时间窗口大小（秒）
    - window_step: 窗口移动步长（秒）
    - freq_threshold: 频率阈值（Hz），只考虑高于此频率的数据点
    """
    print(f"正在加载音频文件: {audio_path}")
    
    # 使用 pydub 加载 MP3 文件
    audio = AudioSegment.from_mp3(audio_path)
    
    # 转换为 numpy 数组
    samples = np.array(audio.get_array_of_samples())
    
    # 如果是立体声，转换为单声道
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    
    # 获取采样率
    sr = audio.frame_rate
    
    # 将采样值标准化到 [-1, 1] 范围
    y = samples.astype(np.float32) / (2**15 if samples.dtype == np.int16 else 2**31)
    
    # 打印音频信息
    duration = len(y) / sr
    print(f"音频采样率: {sr} Hz")
    print(f"音频总长度: {duration:.2f} 秒")
    print(f"音频样本数: {len(y)}")
    
    # 计算总的分析窗口数
    audio_length = duration
    n_windows = max(1, int((audio_length - window_size) / window_step) + 1)
    
    print(f"将分析 {n_windows} 个窗口")
    
    # 首先生成完整音频的聚类
    print("处理完整音频...")
    analyze_window(y, sr, 0, audio_length, freq_threshold, is_full_audio=True)
    
    # 然后按窗口处理
    for window_idx in range(n_windows):
        start_time = window_idx * window_step
        end_time = start_time + window_size
        
        # 确保不超出音频长度
        if end_time > audio_length:
            end_time = audio_length
        
        # 转换为样本索引
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 提取当前窗口的音频
        window_audio = y[start_sample:end_sample]
        
        print(f"处理窗口 {window_idx+1}/{n_windows}: {start_time:.2f}s - {end_time:.2f}s")
        
        # 调用处理函数
        analyze_window(window_audio, sr, start_time, end_time, freq_threshold, window_index=window_idx)

def analyze_window(audio, sr, start_time, end_time, freq_threshold, is_full_audio=False, window_index=None):
    """
    分析单个窗口的音频数据
    
    参数:
    - audio: 窗口内的音频数据
    - sr: 采样率
    - start_time: 窗口开始时间
    - end_time: 窗口结束时间
    - freq_threshold: 频率阈值（Hz）
    - is_full_audio: 是否处理完整音频
    - window_index: 窗口索引，用于生成唯一文件名
    """
    # 修改梅尔频谱图参数
    n_fft = 2048
    hop_length = 512
    n_mels = 256  # 增加梅尔频带数量以提高频率分辨率
    
    # 计算梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels,
        fmin=0,
        fmax=16384  # 设置最大频率范围
    )
    
    # 在提取特征之前应用最小值滤波器
    mel_spec = minimum_filter(mel_spec, size=(12, 6))
    
    # 将梅尔频谱图转换为分贝，并调整对比度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.clip(mel_spec_db, -80, 0)  # 限制动态范围
    
    # 获取梅尔频率值
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=16384)
    
    # 获取时间轴
    times = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sr, hop_length=hop_length)
    
    # 提取特征用于聚类，仅使用振幅（能量强度）
    features = []
    for time_idx in range(mel_spec_db.shape[1]):
        # 特征是时间和振幅
        features.append([
            times[time_idx],      # 时间
            np.mean(mel_spec_db[:, time_idx])  # 振幅（dB），使用时间轴上的平均值
        ])
    
    # 转换为numpy数组
    features = np.array(features)
    
    # 过滤掉振幅很低的点（可能是背景噪音）
    amplitude_threshold = np.median(features[:, 1]) - 15  # 低于中位数15dB的点被忽略
    valid_indices = features[:, 1] > amplitude_threshold
    filtered_features = features[valid_indices]
    
    # 如果没有足够的数据点，则跳过聚类
    if len(filtered_features) < 10:
        print(f"窗口 {start_time:.2f}s - {end_time:.2f}s 中没有足够的高频数据点用于聚类")
        return
    
    # 使用振幅作为聚类特征
    clustering_features = filtered_features[:, 1].reshape(-1, 1)
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_features)
    
    # 使用OPTICS进行聚类，调整参数以更好地分离聚类
    optics = OPTICS(
        min_samples=200,  # 最小样本数
        xi=0.1,
        min_cluster_size=0.1,  # 最小聚类大小
        metric='euclidean',
        max_eps=10 # 设为10减少运算量
    )
    optics.fit(scaled_features)
    
    # 获取可达距离和聚类标签
    reachability = optics.reachability_
    labels = optics.labels_
    ordering = optics.ordering_
    
    # 对可达距离进行归一化处理
    normalized_reachability = normalize_distances(reachability)
    
    # 使用可达距离中位数的3倍作为聚类分界点
    reach_valid = normalized_reachability[np.isfinite(normalized_reachability)]
    if len(reach_valid) > 0:
        reach_median = np.median(reach_valid)
        reach_threshold = reach_median * 3
    else:
        reach_threshold = 0.6  # 如果没有有效的可达距离，使用默认值
    
    # 如果可达距离大于阈值，将其重新标记为噪声点(-1)
    new_labels = np.copy(labels)
    for i in range(len(normalized_reachability)):
        if np.isfinite(normalized_reachability[i]) and normalized_reachability[i] > reach_threshold:
            new_labels[ordering[i]] = -1
    
    # 重新整理标签，使其从0开始连续
    unique_labels = np.unique(new_labels)
    if -1 in unique_labels:
        unique_labels = unique_labels[1:]  # 移除噪声标签
    
    if len(unique_labels) > 0:
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        # 将-1保持为-1（噪声点）
        adjusted_labels = np.array([mapping.get(label, -1) if label != -1 else -1 for label in new_labels])
    else:
        adjusted_labels = new_labels
    
    # 计算总共有多少个聚类
    n_clusters = len(set(adjusted_labels)) - (1 if -1 in adjusted_labels else 0)
    print(f"检测到{n_clusters}个聚类")
    
    # 使用鲜艳的颜色映射，确保聚类之间颜色差异明显
    if n_clusters <= 10:
        cmap = plt.cm.Set1
    else:
        cmap = plt.cm.tab20
    
    colors = cmap(np.linspace(0, 1, max(10, n_clusters)))
    
    # 设置图形大小，为长音频提供更宽的图
    fig_width = 12
    if end_time - start_time > 10:
        fig_width = min(20, 12 + (end_time - start_time) / 2)
    
    plt.figure(figsize=(fig_width, 8))  # 调整图形大小，去掉直方图部分
    
    # 绘制归一化后的可达距离图
    plt.subplot(2, 1, 1)
    
    # 绘制可达距离，使用渐变色以便更好地识别
    for i in range(len(ordering)):
        label = adjusted_labels[ordering[i]]
        if label == -1:
            color = 'lightgray'  # 使用浅灰色表示噪声
        else:
            color = colors[label % len(colors)]
        
        # 使用稍透明的条形图
        plt.bar(i, normalized_reachability[ordering[i]], color=color, width=1.0, alpha=0.8)
    
    # 在图上标记阈值
    plt.axhline(y=reach_threshold, color='r', linestyle='-', linewidth=2, label=f'阈值 ({reach_threshold:.2f})')
    
    # 添加标题和标签
    window_info = f" (窗口: {start_time:.2f}s - {end_time:.2f}s)" if not is_full_audio else " (完整音频)"
    plt.title(f'OPTICS归一化可达距离图{window_info}', fontsize=14)
    plt.xlabel('样本点 (按聚类顺序)', fontsize=12)
    plt.ylabel('归一化可达距离', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 绘制带有聚类标记的梅尔谱图
    plt.subplot(2, 1, 2)
    
    # 使用原始梅尔谱图作为背景
    plt.imshow(mel_spec_db, 
              aspect='auto', 
              origin='lower', 
              cmap='magma',  # 使用magma颜色映射
              alpha=0.3,
              extent=[0, times[-1], 0, mel_freqs[-1]])
    
    # 创建一个新的梅尔谱图，每个点根据其聚类标签着色
    mel_img = np.zeros((mel_spec_db.shape[0], mel_spec_db.shape[1], 4))
    
    # 为每个频率-时间点绘制聚类
    for i, feature in enumerate(filtered_features):
        # 计算对应的梅尔频率和时间索引
        freq_indices = np.where(np.isclose(mel_freqs, feature[0]))[0]
        
        if len(freq_indices) > 0:
            freq_idx = freq_indices[0]
        else:
            print(f"警告：未找到频率 {feature[0]} 的匹配索引")
            continue
        
        time_indices = np.where(np.isclose(times, feature[1]))[0]
        
        if len(time_indices) > 0:
            time_idx = time_indices[0]
        else:
            print(f"警告：未找到时间 {feature[1]} 的匹配索引")
            continue
        
        # 根据标签设置颜色
        label = adjusted_labels[i]
        if label == -1:
            color = [0.5, 0.5, 0.5, 0.5]  # 透明灰色
        else:
            rgb_color = colors[label % len(colors)][:3]
            color = [*rgb_color, 0.8]  # 加透明度
        
        # 只绘制能量高于阈值的点
        if feature[2] > amplitude_threshold:
            mel_img[freq_idx, time_idx] = color
    
    # 显示聚类后的梅尔谱图
    plt.imshow(mel_img, aspect='auto', origin='lower', 
              extent=[0, times[-1], 0, mel_freqs[-1]])
    
    # 假设 freq_threshold 是频率阈值
    mel_threshold_idx = np.where(mel_freqs >= freq_threshold)[0][0] if np.any(mel_freqs >= freq_threshold) else -1

    # 绘制分割线指示高频区域的开始
    if mel_threshold_idx != -1:
        plt.axhline(y=mel_freqs[mel_threshold_idx], color='r', linestyle=':', linewidth=2, 
                    label=f'频率阈值 ({freq_threshold} Hz)')
    
    # 设置标题和标签
    plt.title(f'聚类后的梅尔谱图{window_info}', fontsize=14)
    plt.ylabel('梅尔频率 (Hz)', fontsize=12)
    plt.xlabel('时间 (秒)', fontsize=12)
    
    # 使用对数刻度来显示梅尔频率轴
    plt.yscale('symlog')
    
    # 添加颜色条来显示聚类
    unique_labels_for_legend = np.unique(adjusted_labels)
    handles = []
    labels_text = []
    
    for label in unique_labels_for_legend:
        if label == -1:
            handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10)
            labels_text.append('噪声')
        else:
            handle = plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=colors[label % len(colors)], markersize=10)
            labels_text.append(f'聚类 {label}')
        handles.append(handle)
    
    plt.legend(handles, labels_text, loc='upper right', fontsize=10)
    
    # 添加网格线以便更好地定位
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存结果
    plt.tight_layout()
    
    # 生成唯一文件名
    file_suffix = f"{TIMESTAMP}"
    if window_index is not None:
        file_suffix += f"_win{window_index}"
    
    # 生成文件名
    if is_full_audio:
        output_filename = os.path.join(OUTPUT_DIR, f'OPTICS聚类_完整音频_{file_suffix}.png')
    else:
        output_filename = os.path.join(OUTPUT_DIR, f'OPTICS聚类_{start_time:.2f}s-{end_time:.2f}s_{file_suffix}.png')
    
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"已保存聚类结果到 {output_filename}")

if __name__ == "__main__":
    # 指定音频文件路径
    audio_file = "原始音频片段.mp3"
    
    # 调用聚类分析函数，处理完整音频和窗口
    mel_spectrogram_with_optics_clustering(audio_file, window_size=3, window_step=1.5, freq_threshold=3500)
    
    print("聚类分析完成，结果保存在", OUTPUT_DIR)
