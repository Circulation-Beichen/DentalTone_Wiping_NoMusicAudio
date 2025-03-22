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
import soundfile as sf
from scipy import signal
import torch
import concurrent.futures
warnings.filterwarnings('ignore')

# 设置 matplotlib 使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置输出目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "OPTICS聚类结果")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 生成唯一时间戳
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# 检查 GPU 并设置设备
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
print(f"使用{'GPU' if HAS_GPU else 'CPU'}运行 OPTICS 聚类算法")

# CPU 线程数（例如 i9-13900H 有 24 线程）
N_THREADS = 24

def load_audio(audio_path):
    """加载音频文件"""
    print(f"正在加载音频: {audio_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    print(f"音频加载成功! 采样率: {sr}Hz, 长度: {len(y)/sr:.2f}秒")
    return y, sr

def parallel_stft(audio, sr, n_blocks=N_THREADS):
    """并行计算 STFT（CPU）"""
    block_size = len(audio) // n_blocks
    blocks = [audio[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_blocks) as executor:
        stft_blocks = list(executor.map(lambda x: librosa.stft(x, n_fft=2048, hop_length=512), blocks))
    
    return np.hstack(stft_blocks)

def gpu_stft(audio, sr):
    """使用 GPU 计算 STFT"""
    audio_tensor = torch.from_numpy(audio).float().to(DEVICE)
    D = torch.stft(audio_tensor, n_fft=2048, hop_length=512, return_complex=True)
    return D

def preprocess_audio(audio, sr, threshold_db=-20, width=7, intensity=0.7, order=1):
    """预处理音频"""
    # 计算 STFT
    if HAS_GPU:
        D = gpu_stft(audio, sr)
        magnitude = torch.abs(D)
        phase = torch.angle(D)
    else:
        D = parallel_stft(audio, sr)
        magnitude = torch.from_numpy(np.abs(D)).to(DEVICE)
        phase = torch.from_numpy(np.angle(D)).to(DEVICE)
    
    # 频率和时间轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr)
    
    # 删除 3kHz 以下频率
    low_freq_mask = torch.tensor(freqs < 3000, device=DEVICE)
    filtered_magnitude = magnitude.clone()
    filtered_magnitude[low_freq_mask, :] = 0
    
    # 删除低于阈值的幅度
    magnitude_db = 20 * torch.log10(filtered_magnitude + 1e-10)
    low_amp_mask = magnitude_db < threshold_db
    filtered_magnitude[low_amp_mask] = 0
    
    # 应用微分滤波器
    if order == 1:
        diff_kernel = torch.ones(width, device=DEVICE) / width
        diff_kernel[width//2:] *= -1
    else:
        diff_kernel = torch.zeros(width, device=DEVICE)
        if width >= 3:
            center = width // 2
            diff_kernel[center] = 2 / width
            if center > 0:
                diff_kernel[center-1] = -1 / width
            if center < width - 1:
                diff_kernel[center+1] = -1 / width
    
    filtered_mag = filtered_magnitude.clone()
    for f_idx in range(filtered_mag.shape[0]):
        if not (low_freq_mask[f_idx] or torch.all(filtered_mag[f_idx, :] == 0)):
            filtered_row = torch.conv1d(
                filtered_mag[f_idx:f_idx+1, :], 
                diff_kernel.view(1, 1, -1), 
                padding=(width-1)//2
            )[0, 0]
            filtered_mag[f_idx, :] = filtered_mag[f_idx, :] * (1-intensity) + filtered_row * intensity
    
    # 转换为 dB 并生成掩码
    filtered_magnitude_db = 20 * torch.log10(filtered_mag + 1e-10)
    non_zero_mask = filtered_mag > 0
    
    return filtered_mag, phase, non_zero_mask, magnitude_db, filtered_magnitude_db, freqs, times

def extract_features_from_magnitude(filtered_mag, freqs, times):
    """提取特征用于聚类"""
    filtered_magnitude_db = 20 * torch.log10(filtered_mag + 1e-10)
    non_zero_mask = filtered_mag > 0
    t_grid, f_grid = torch.meshgrid(torch.tensor(times, device=DEVICE), torch.tensor(freqs, device=DEVICE), indexing='ij')
    features = torch.stack([
        t_grid.T[non_zero_mask].float(),
        f_grid.T[non_zero_mask].float(),
        filtered_magnitude_db[non_zero_mask].float()
    ], dim=1)
    return features.cpu().numpy()

def gpu_distance_matrix(X, time_scale, mel_scale, time_window, mel_window, max_eps, batch_size=1000):
    """使用 GPU 计算距离矩阵（批处理）"""
    n_samples = len(X)
    distance_matrix = np.full((n_samples, n_samples), np.inf)
    
    X_tensor = torch.from_numpy(X[:, :2]).float().to(DEVICE)
    mel_freqs = torch.tensor([librosa.hz_to_mel(x[1]) for x in X], device=DEVICE)
    
    for i in range(0, n_samples, batch_size):
        start_i = i
        end_i = min(i + batch_size, n_samples)
        
        time_diff_i = torch.abs(X_tensor[start_i:end_i, 0].unsqueeze(1) - X_tensor[:, 0].unsqueeze(0))
        mel_diff_i = torch.abs(mel_freqs[start_i:end_i].unsqueeze(1) - mel_freqs.unsqueeze(0))
        
        mask_i = (time_diff_i <= time_window) & (mel_diff_i <= mel_window)
        dist_i = torch.where(
            mask_i,
            torch.maximum(time_diff_i * time_scale, mel_diff_i * mel_scale),
            torch.tensor(float('inf'), device=DEVICE)
        )
        dist_i = torch.where(dist_i <= max_eps, dist_i, torch.tensor(float('inf'), device=DEVICE))
        
        distance_matrix[start_i:end_i, :] = dist_i.cpu().numpy()
        distance_matrix[:, start_i:end_i] = dist_i.T.cpu().numpy()  # 确保对称
    
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def cpu_distance_matrix(X, time_scale, mel_scale, time_window, mel_window, max_eps):
    """使用 CPU 计算距离矩阵"""
    n_samples = len(X)
    distance_matrix = np.full((n_samples, n_samples), np.inf)
    
    times = X[:, 0]
    mel_freqs = np.array([librosa.hz_to_mel(x[1]) for x in X])
    
    for i in range(n_samples):
        time_diff = np.abs(times[i] - times)
        mel_diff = np.abs(mel_freqs[i] - mel_freqs)
        mask = (time_diff <= time_window) & (mel_diff <= mel_window)
        dist = np.maximum(time_diff * time_scale, mel_diff * mel_scale)
        dist = np.where(dist <= max_eps, dist, np.inf)
        distance_matrix[i, :] = dist
        distance_matrix[:, i] = dist  # 确保对称
    
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def perform_optics_clustering(features, time_scale, mel_scale, max_eps=None, min_samples=10, batch_size=1000):
    """执行 OPTICS 聚类"""
    if len(features) < min_samples:
        print(f"没有足够的数据点用于聚类")
        return np.full(len(features), -1)

    X = features[:, :2]
    times = X[:, 0]
    freqs = X[:, 1]

    min_time_interval = np.min(np.diff(np.sort(times))[np.diff(np.sort(times)) > 0], initial=0.01)
    min_freq_interval = np.min(np.diff(np.sort(freqs))[np.diff(np.sort(freqs)) > 0], initial=1.0)
    max_eps = max_eps or min(50 * min_time_interval * time_scale, 200.0 * mel_scale)
    time_window = 50 * min_time_interval
    mel_window = 200.0

    print(f"最大可达距离: {max_eps:.6f}, 时间窗口: {time_window:.6f}秒, 梅尔窗口: {mel_window:.2f} mel")
    print(f"数据点总数: {len(X)}")

    if HAS_GPU:
        distance_matrix = gpu_distance_matrix(X, time_scale, mel_scale, time_window, mel_window, max_eps, batch_size=batch_size)
    else:
        distance_matrix = cpu_distance_matrix(X, time_scale, mel_scale, time_window, mel_window, max_eps)

    optics = OPTICS(
        min_samples=min_samples,
        metric='precomputed',
        max_eps=max_eps,
        cluster_method='xi',
        xi=0.05
    )
    optics.fit(distance_matrix)
    labels = optics.labels_

    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    print(f"聚类结果: {n_clusters}个类别，噪声点: {np.sum(labels == -1)}/{len(X)}")
    return labels

def filter_clusters_by_frequency_span(features, labels):
    """过滤频率跨度小的聚类"""
    filtered_labels = labels.copy()
    unique_labels = np.unique(labels)
    max_freq_span = 0
    cluster_freq_spans = {}

    for label in unique_labels:
        if label != -1:
            cluster_points = features[labels == label]
            freq_span = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            cluster_freq_spans[label] = freq_span
            max_freq_span = max(max_freq_span, freq_span)

    threshold_span = max_freq_span / 2
    print(f"最大频率跨度: {max_freq_span:.2f} Hz, 阈值: {threshold_span:.2f} Hz")

    cluster_centers = []
    for label in unique_labels:
        if label != -1 and cluster_freq_spans[label] >= threshold_span:
            cluster_points = features[labels == label]
            weights = 10 ** (cluster_points[:, 2] / 20)
            center = np.sum(cluster_points[:, :2] * weights[:, np.newaxis], axis=0) / np.sum(weights)
            cluster_centers.append(center)
        elif label != -1:
            filtered_labels[labels == label] = -1

    return filtered_labels, cluster_centers

def filter_clusters_by_min_frequency_span(features, labels, min_span=6000):
    """过滤频率跨度小于 min_span 的聚类"""
    filtered_labels = labels.copy()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label != -1:
            cluster_points = features[labels == label]
            freq_span = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            if freq_span < min_span:
                filtered_labels[labels == label] = -1

    return filtered_labels

def apply_gain_reduction(audio, sr, features, labels, phase, max_reduction_db=20):
    """在 GPU 上应用负增益"""
    D = torch.from_numpy(librosa.stft(audio)).to(DEVICE)
    magnitude = torch.abs(D)
    gain_factors = torch.ones_like(magnitude)

    valid_points = features[labels != -1]
    if len(valid_points) > 0:
        times = torch.tensor(librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr), device=DEVICE)
        freqs = torch.tensor(librosa.fft_frequencies(sr=sr, n_fft=2048), device=DEVICE)
        valid_points_tensor = torch.from_numpy(valid_points).to(DEVICE)

        t_indices = torch.argmin(torch.abs(times.unsqueeze(0) - valid_points_tensor[:, 0].unsqueeze(1)), dim=1)
        f_indices = torch.argmin(torch.abs(freqs.unsqueeze(0) - valid_points_tensor[:, 1].unsqueeze(1)), dim=1)

        amplitudes = valid_points_tensor[:, 2]
        norm_amplitudes = (amplitudes - amplitudes.min()) / (amplitudes.max() - amplitudes.min() + 1e-10)
        reductions = max_reduction_db * norm_amplitudes
        gain_factors[f_indices, t_indices] = 10 ** (-reductions / 20)

    processed_magnitude = magnitude * gain_factors
    processed_D = processed_magnitude * torch.exp(1j * phase)
    processed_audio = librosa.istft(processed_D.cpu().numpy())
    return processed_audio, gain_factors.cpu().numpy()

def visualize_clustering_results(features, labels, clusters_name, output_path):
    """可视化聚类结果"""
    if len(features) == 0:
        print(f"没有数据可供可视化")
        return

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"检测到 {n_clusters} 个聚类")

    plt.figure(figsize=(12, 8))
    cmap = plt.cm.tab10 if n_clusters <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(10, n_clusters)))

    noise_points = features[labels == -1]
    if len(noise_points) > 0:
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='lightgray', marker='.', s=30, alpha=0.5, label='噪声')

    for i, label in enumerate(unique_labels):
        if label != -1:
            cluster_points = features[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i % len(colors)]], marker='o', s=50, alpha=0.8, label=f'聚类 {label}')
            center = np.mean(cluster_points[:, :2], axis=0)
            plt.scatter(center[0], center[1], c=[colors[i % len(colors)]], marker='x', s=200, linewidths=3, edgecolors='black')

    plt.title(f'{clusters_name} 聚类结果', fontsize=16)
    plt.xlabel('时间 (秒)', fontsize=14)
    plt.ylabel('频率 (Hz)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    if n_clusters > 0:
        plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存聚类结果到 {output_path}")

def main():
    audio_file = "原始音频片段_01.mp3"
    audio_path = os.path.join(ROOT_DIR, audio_file)
    output_dir = OUTPUT_DIR

    audio, sr = load_audio(audio_path)

    print("第一次预处理：使用参数 (-20dB 阈值，w=7，i=0.7，阶=1)")
    filtered_mag1, phase1, non_zero_mask1, magnitude_db1, filtered_magnitude_db1, freqs1, times1 = preprocess_audio(
        audio, sr, threshold_db=-20, width=7, intensity=0.7, order=1
    )

    features1 = extract_features_from_magnitude(filtered_mag1, freqs1, times1)
    time_scale = 1.0
    mel_scale = 1.0

    print("第一次聚类：使用梅尔尺度的棋盘距离进行 OPTICS 聚类")
    labels1 = perform_optics_clustering(features1, time_scale, mel_scale, min_samples=5, batch_size=1000)

    print("过滤第一次聚类，保留频率跨度不小于最大跨度二分之一的聚类")
    filtered_labels1, cluster_centers1 = filter_clusters_by_frequency_span(features1, labels1)

    print("可视化第一次聚类结果")
    visualize_clustering_results(
        features1, 
        filtered_labels1, 
        "第一次", 
        os.path.join(output_dir, f"第一次聚类结果_{TIMESTAMP}.png")
    )

    print("第二次预处理：使用参数 (-30dB 阈值，w=7，i=0.5，阶=2)")
    filtered_mag2, phase2, non_zero_mask2, magnitude_db2, filtered_magnitude_db2, freqs2, times2 = preprocess_audio(
        audio, sr, threshold_db=-30, width=7, intensity=0.5, order=2
    )

    features2 = extract_features_from_magnitude(filtered_mag2, freqs2, times2)

    print("第二次聚类：使用梅尔尺度的棋盘距离进行 OPTICS 聚类")
    labels2 = perform_optics_clustering(features2, time_scale, mel_scale, min_samples=5, batch_size=1000)

    print("过滤第二次聚类，仅保留频率跨度大于 6kHz 的聚类")
    filtered_labels2 = filter_clusters_by_min_frequency_span(features2, labels2, min_span=6000)

    print("可视化第二次聚类结果")
    visualize_clustering_results(
        features2, 
        filtered_labels2, 
        "第二次", 
        os.path.join(output_dir, f"第二次聚类结果_{TIMESTAMP}.png")
    )

    print("对原音频应用基于聚类的负增益")
    processed_audio, gain_factors = apply_gain_reduction(
        audio, sr, features2, filtered_labels2, phase2, max_reduction_db=20
    )

    output_audio_path = os.path.join(output_dir, f"处理后音频_{TIMESTAMP}.wav")
    sf.write(output_audio_path, processed_audio, sr)
    print(f"处理后音频已保存至: {output_audio_path}")

    with open(os.path.join(output_dir, f"处理说明_{TIMESTAMP}.txt"), 'w', encoding='utf-8') as f:
        f.write("OPTICS 聚类处理说明\n")
        f.write("="*30 + "\n\n")
        f.write(f"处理参数:\n")
        f.write(f"- 音频文件: {audio_file}\n")
        f.write(f"- 第一次预处理: 阈值=-20dB, 滤波器宽度=7, 强度=0.7, 阶数=1\n")
        f.write(f"- 第二次预处理: 阈值=-30dB, 滤波器宽度=7, 强度=0.5, 阶数=2\n")
        f.write(f"- 聚类距离: 梅尔尺度下的棋盘距离\n")
        f.write(f"- 时间窗口: 50 个最小时间间隔\n")
        f.write(f"- 梅尔窗口: 200 个 mel 单位\n")
        f.write(f"- 聚类过滤: 频率跨度 > 6kHz\n")
        f.write(f"- 最大负增益: 20dB\n")
        f.write(f"- 计算设备: {'GPU' if HAS_GPU else 'CPU'}\n")
        f.write(f"- CPU 线程数: {N_THREADS}\n\n")
        f.write("处理步骤:\n")
        f.write("1. 使用 (-20dB 阈值，w=7，i=0.7，阶=1) 预处理音频\n")
        f.write("2. 使用 OPTICS 算法对预处理结果进行聚类，使用梅尔尺度的棋盘距离\n")
        f.write("3. 保留频率跨度不小于最大跨度二分之一的聚类\n")
        f.write("4. 使用 (-30dB 阈值，w=7，i=0.5，阶=2) 再次预处理音频\n")
        f.write("5. 使用 OPTICS 算法对第二次预处理结果进行聚类\n")
        f.write("6. 仅保留频率跨度大于 6kHz 的聚类\n")
        f.write("7. 对原音频应用负增益，聚类中幅度越大的点负增益越大（最大 -20dB）\n\n")
        f.write("优化措施:\n")
        f.write("1. STFT 计算: GPU 加速（torch.stft）或 CPU 多线程\n")
        f.write("2. 距离矩阵: GPU 向量化计算（批处理）\n")
        f.write("3. 增益应用: GPU 向量化操作\n")
        f.write("4. 内存使用: 缓存所有中间结果，充分利用额外 20GB 内存\n\n")
        f.write("处理结果:\n")
        f.write(f"- 第一次聚类结果: {os.path.join(output_dir, f'第一次聚类结果_{TIMESTAMP}.png')}\n")
        f.write(f"- 第二次聚类结果: {os.path.join(output_dir, f'第二次聚类结果_{TIMESTAMP}.png')}\n")
        f.write(f"- 处理后音频: {output_audio_path}\n")

    print(f"处理完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()