import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import librosa.display
import os
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

def plot_spectrogram(y, sr, title, filename):
    """生成并保存频谱图"""
    plt.figure(figsize=(12, 6))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    
    # 计算梅尔频谱图
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # 绘制频谱图
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    
    # 保存图片
    plt.tight_layout()
    output_path = os.path.join(ROOT_DIR, filename)
    plt.savefig(output_path, dpi=300)
    print(f"频谱图已保存到: {output_path}")
    plt.close()

def extract_audio_segment(audio_path, start_sec, end_sec):
    """提取音频片段"""
    # 确保使用绝对路径
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(ROOT_DIR, audio_path)
    
    print(f"尝试加载音频文件: {audio_path}")
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件 '{audio_path}' 不存在")
        print(f"当前工作目录: {ROOT_DIR}")
        print("目录中的文件:")
        for file in os.listdir(ROOT_DIR):
            if file.endswith(('.mp3', '.wav')):
                print(f"  - {file}")
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    
    # 如果是mp3格式，使用pydub提取片段
    if audio_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
        segment = audio[start_sec*1000:end_sec*1000]
        # 临时保存为wav格式
        temp_wav = os.path.join(ROOT_DIR, "temp_segment.wav")
        segment.export(temp_wav, format="wav")
        # 使用librosa加载
        y, sr = librosa.load(temp_wav, sr=None)
        # 删除临时文件
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    else:
        # 直接使用librosa加载完整音频
        y, sr = librosa.load(audio_path, sr=None)
        # 提取指定时间段
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        y = y[start_sample:end_sample]
    
    print(f"音频加载成功! 采样率: {sr}Hz, 长度: {len(y)/sr:.2f}秒")
    return y, sr

def save_as_mp3(audio, sr, filename):
    """保存为mp3格式"""
    # 确保使用绝对路径
    output_path = os.path.join(ROOT_DIR, filename)
    
    # 先保存为wav格式
    temp_wav = os.path.join(ROOT_DIR, "temp_output.wav")
    sf.write(temp_wav, audio, sr)
    
    # 转换为mp3格式
    audio_segment = AudioSegment.from_wav(temp_wav)
    audio_segment.export(output_path, format="mp3")
    print(f"音频已保存到: {output_path}")
    
    # 删除临时wav文件
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

def extract_features(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """从音频信号中提取特征用于弦乐识别"""
    # MFCC特征 - 音色和音高特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)  # MFCC的一阶导数
    
    # 频谱质心 - 描述声音的"亮度"
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 光谱对比度 - 描述高频和低频之间的差异
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 色谱图 - 音高特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 零交叉率 - 有助于区分持续音调和噪声
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # 组合所有特征并计算统计量
    features = []
    feature_list = [mfcc, mfcc_delta, spectral_centroid, spectral_contrast, chroma, zcr]
    
    for feature in feature_list:
        features.extend([
            np.mean(feature),
            np.std(feature),
            np.min(feature),
            np.max(feature),
            np.median(feature),
        ])
    
    return np.array(features)

def train_string_instrument_classifier(audio_path, segment_duration=3, overlap=1.5):
    """
    使用卡拉扬_贝多芬第五交响乐的音频训练弦乐识别模型
    :param audio_path: 音频文件路径
    :param segment_duration: 每个训练样本的持续时间(秒)
    :param overlap: 相邻样本之间的重叠时间(秒)
    """
    print(f"开始训练弦乐识别模型，使用素材: {audio_path}")
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None)
    
    # 分割音频为多个片段用于训练
    samples_per_segment = int(segment_duration * sr)
    samples_per_overlap = int(overlap * sr)
    samples_per_hop = samples_per_segment - samples_per_overlap
    
    # 存储特征和标签
    features = []
    
    # 手动标记：这里我们简化为二分类问题
    # 1. 有弦乐主导的片段 (标签1)
    # 2. 无弦乐或弦乐不明显的片段 (标签0)
    # 注意：在真实应用中，需要手动标记或使用另一个分类器预先识别
    
    # 为简化示例，我们假设前1/3的音频为非弦乐主导，后2/3为弦乐主导
    segment_count = (len(y) - samples_per_segment) // samples_per_hop + 1
    one_third_segments = segment_count // 3
    
    labels = []
    segments = []
    
    # 分割并提取特征
    for i in range(segment_count):
        start = i * samples_per_hop
        end = start + samples_per_segment
        
        if end <= len(y):
            segment = y[start:end]
            segments.append(segment)
            
            # 提取特征
            segment_features = extract_features(segment, sr)
            features.append(segment_features)
            
            # 分配标签（这里是简化的标记，实际应用中应使用正确的标记）
            if i < one_third_segments:
                labels.append(0)  # 非弦乐主导
            else:
                labels.append(1)  # 弦乐主导
    
    # 转换为numpy数组
    X = np.array(features)
    y = np.array(labels)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 评估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['非弦乐主导', '弦乐主导'])
    
    print(f"模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    # 保存模型
    model_path = os.path.join(ROOT_DIR, "string_instrument_classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 绘制一些弦乐和非弦乐片段的频谱图作为参考
    for i, label_name in enumerate(['非弦乐主导', '弦乐主导']):
        # 找到对应标签的前两个样本
        indices = [idx for idx, label in enumerate(labels) if label == i][:2]
        
        for j, idx in enumerate(indices):
            # 绘制频谱图
            segment = segments[idx]
            plot_spectrogram(segment, sr, f"{label_name}频谱示例 #{j+1}", f"{label_name}_示例_{j+1}.png")
    
    return clf, accuracy

def weighted_high_freq_average(audio, sr, freq_min=4000, freq_max=20000, peak_freq=6500):
    """
    计算音频的加权高频平均值，5k-8kHz权重最高
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
    
    # 创建权重，5k-8kHz处权重为3，向两侧递减到1
    weights = np.ones_like(mel_freqs)
    
    for i, freq in enumerate(mel_freqs):
        # 距离峰值频率的距离，并标准化到0-1范围
        dist = min(abs(freq - peak_freq) / (peak_freq - freq_min), 1.0)
        # 根据距离计算权重，最大为3，最小为1
        weights[i] = max(3 * (1 - dist), 1.0)
    
    # 应用权重
    weighted_mel = mel_spec * weights[:, np.newaxis]
    
    # 计算加权平均值
    weighted_avg = np.mean(weighted_mel, axis=0)
    
    return weighted_avg, mel_spec, weights

def improved_string_extraction(audio, sr):
    """
    改进的弦乐提取方法，使用频谱特征和谐波跟踪
    :param audio: 输入音频
    :param sr: 采样率
    :return: 提取的弦乐音频
    """
    # 1. 频谱分析
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 2. 谐波增强 - 弦乐有丰富的谐波结构
    harmonic = librosa.effects.harmonic(audio, margin=8.0)
    
    # 3. 计算谐波-打击乐分离
    D_harmonic = librosa.stft(harmonic, n_fft=2048, hop_length=512)
    
    # 4. 提取谐波部分的特征
    # 谱平滑度 - 弦乐较平滑
    spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
    
    # 计算频谱对比度 - 弦乐有显著的谐波峰值
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
    
    # 5. 创建弦乐掩码
    # 低平滑度且高对比度的区域更可能是弦乐
    
    # 打印维度信息用于调试
    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Spectral flatness shape: {spectral_flatness.shape}")
    
    # spectral_flatness是一个(1, n_frames)形状的数组，我们需要改变处理方式
    
    # 首先，将spectral_flatness调整为一维数组
    flatness_values = spectral_flatness[0]  # 变为(n_frames,)
    
    # 设置阈值
    flatness_threshold = np.percentile(flatness_values, 70)
    
    # 创建一个(n_freqs, n_frames)形状的初始掩码矩阵
    mask = np.zeros_like(magnitude, dtype=bool)
    
    # 基于时间帧的平滑度创建掩码
    # 对每一个时间帧，所有频率点使用相同的掩码值
    for i in range(magnitude.shape[1]):
        # 如果该帧的平滑度低于阈值（可能是弦乐），则在掩码中标记为True
        if flatness_values[i] < flatness_threshold:
            mask[:, i] = True
    
    # 谐波增强参数
    harmonic_gain = 1.5  # 提升谐波成分
    percussive_reduction = 0.3  # 降低打击乐成分
    
    # 应用掩码和谐波增强
    enhanced_magnitude = np.where(
        mask, 
        magnitude * harmonic_gain,  # 谐波增强
        magnitude * percussive_reduction  # 打击乐降低
    )
    
    # 保留低于2000Hz的成分，弦乐主要集中在这一区域
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    for i, freq in enumerate(freqs):
        if freq > 2000:
            factor = np.exp(-0.001 * (freq - 2000))  # 逐渐衰减高频
            enhanced_magnitude[i, :] *= factor
    
    # 重建音频
    y_enhanced = librosa.istft(enhanced_magnitude * np.exp(1j * phase))
    
    # 6. 应用额外的谐波滤波器
    y_enhanced = librosa.effects.harmonic(y_enhanced)
    
    # 7. 标准化
    y_enhanced = y_enhanced / np.max(np.abs(y_enhanced))
    
    return y_enhanced

def detect_high_freq_clusters_with_weights(audio, sr, time_start=1, time_end=3, freq_threshold=4000, gain_db=-20, n_clusters=3):
    """
    使用加权Mel频率特征的DPCA算法检测并处理指定时间范围内的高频聚类
    :param audio: 输入音频信号
    :param sr: 采样率
    :param time_start: 起始时间(秒)
    :param time_end: 结束时间(秒)
    :param freq_threshold: 频率阈值，只处理高于此频率的成分(Hz)
    :param gain_db: 应用于聚类的增益(dB)
    :param n_clusters: 聚类数量
    :return: 处理后的音频
    """
    # 将时间转换为样本索引
    start_sample = int(time_start * sr)
    end_sample = int(time_end * sr)
    
    # 提取目标时间范围内的音频片段
    target_segment = audio[start_sample:end_sample]
    
    # 计算加权高频平均值
    weighted_avg, mel_spec, weights = weighted_high_freq_average(
        target_segment, sr, freq_min=4000, freq_max=20000, peak_freq=6500
    )
    
    # 绘制加权高频平均值
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
    plt.savefig(os.path.join(ROOT_DIR, "高频加权分析.png"))
    plt.close()
    
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
    
    # 实现密度峰值聚类 (DPCA)
    # 1. 计算每对点之间的距离矩阵
    num_points = features_scaled.shape[0]
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            dist = np.sqrt(np.sum((features_scaled[i] - features_scaled[j])**2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # 2. 计算每个点的局部密度 rho
    dc = np.percentile(dist_matrix[dist_matrix > 0], 20)  # 距离截断值设为所有距离的20%分位数
    rho = np.zeros(num_points)
    for i in range(num_points):
        # 计算高斯密度
        rho[i] = np.sum(np.exp(-(dist_matrix[i] / dc)**2))
    
    # 3. 对于每个点，找到距离它最近的且密度更高的点的距离
    delta = np.ones(num_points) * np.max(dist_matrix)
    nearest_neighbor = np.ones(num_points, dtype=int) * -1
    
    # 对点按密度排序
    rho_order = np.argsort(-rho)
    
    for i in range(1, num_points):
        idx = rho_order[i]
        for j in range(i):
            prev_idx = rho_order[j]
            if dist_matrix[idx, prev_idx] < delta[idx]:
                delta[idx] = dist_matrix[idx, prev_idx]
                nearest_neighbor[idx] = prev_idx
    
    # 对密度最高的点，delta是最大的距离
    delta[rho_order[0]] = np.max(delta)
    
    # 4. 确定聚类中心：高密度和高delta的点
    # 计算gamma值
    gamma = rho * delta
    cluster_centers_idx = np.argsort(-gamma)[:n_clusters]
    
    # 可视化决策图
    plt.figure(figsize=(10, 8))
    plt.scatter(rho, delta, s=20, c='k', alpha=0.5)
    plt.scatter(rho[cluster_centers_idx], delta[cluster_centers_idx], s=80, c='r', marker='*')
    for i, idx in enumerate(cluster_centers_idx):
        plt.annotate(f'聚类中心 #{i+1}', (rho[idx], delta[idx]))
    plt.xlabel('局部密度 (rho)')
    plt.ylabel('距离 (delta)')
    plt.title('DPCA 聚类决策图')
    plt.savefig(os.path.join(ROOT_DIR, "DPCA_聚类决策图.png"))
    plt.close()
    
    # 5. 分配每个点到最近的聚类中心
    cluster_assignments = np.zeros(num_points, dtype=int) - 1
    
    # 首先分配聚类中心
    for i, idx in enumerate(cluster_centers_idx):
        cluster_assignments[idx] = i
    
    # 按密度从高到低分配其余点
    for i in range(num_points):
        idx = rho_order[i]
        if cluster_assignments[idx] == -1:  # 如果尚未分配
            if nearest_neighbor[idx] != -1:  # 确保有邻居
                cluster_assignments[idx] = cluster_assignments[nearest_neighbor[idx]]
    
    # 找出每个聚类中能量最高的时间帧
    cluster_centers = []
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_assignments == i)[0]
        if len(cluster_indices) > 0:
            # 找出聚类中加权能量最高的时间帧
            energy_scores = time_features_array[cluster_indices, 1] * time_features_array[cluster_indices, 2]  # 幅度 * 加权值
            max_energy_idx = cluster_indices[np.argmax(energy_scores)]
            t_time = time_features_array[max_energy_idx, 0]  # 时间
            
            # 转换为STFT帧索引
            t_idx = np.argmin(np.abs(times - t_time))
            
            # 记录聚类中心的时间帧索引和平均能量
            avg_energy = time_features_array[max_energy_idx, 1]
            cluster_centers.append((t_idx, avg_energy))
    
    print(f"检测到的{len(cluster_centers)}个高频时间聚类中心:")
    for i, (t_idx, avg_energy) in enumerate(cluster_centers):
        print(f"聚类中心 #{i+1}: 时间={times[t_idx]:.2f}秒, 平均高频能量={librosa.amplitude_to_db(avg_energy):.1f}dB")
    
    # 创建处理后的STFT矩阵
    processed_magnitude = magnitude.copy()
    
    # 在每个聚类中心时间的整个频率范围上应用衰减，但只对高频部分
    for t_idx, _ in cluster_centers:
        # 定义时间窗口范围
        t_window = 8  # 时间范围窗口大小
        
        # 安全地获取周围区域
        t_start = max(0, t_idx - t_window)
        t_end = min(processed_magnitude.shape[1], t_idx + t_window + 1)
        
        # 只对高频部分应用增益
        for f_idx in high_freq_indices:
            # 应用增益
            gain_factor = 10 ** (gain_db / 20)  # 将dB转换为线性增益因子
            
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

def extract_string_instruments(audio, sr, model_path=None):
    """
    从音频中提取弦乐部分
    :param audio: 输入音频信号
    :param sr: 采样率
    :param model_path: 训练好的弦乐识别模型路径
    :return: 仅包含弦乐的音频信号
    """
    if model_path is None:
        model_path = os.path.join(ROOT_DIR, "string_instrument_classifier.joblib")
    
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        print("请先训练模型")
        return audio
    
    # 加载模型
    clf = joblib.load(model_path)
    
    # 分割音频为小段进行处理
    segment_duration = 0.5  # 秒
    hop_length_duration = 0.25  # 秒
    
    samples_per_segment = int(segment_duration * sr)
    hop_length_samples = int(hop_length_duration * sr)
    
    # 创建相同长度的输出音频，初始化为零
    output_audio = np.zeros_like(audio)
    
    # 使用窗口函数进行平滑过渡
    window = np.hanning(samples_per_segment)
    
    # 逐段处理
    for start_sample in range(0, len(audio) - samples_per_segment, hop_length_samples):
        end_sample = start_sample + samples_per_segment
        segment = audio[start_sample:end_sample]
        
        # 提取特征
        features = extract_features(segment, sr).reshape(1, -1)
        
        # 预测当前段是否包含弦乐
        prediction = clf.predict(features)[0]
        
        # 如果预测为弦乐，则保留该段
        if prediction == 1:  # 弦乐主导
            # 应用窗口函数使过渡平滑
            output_audio[start_sample:end_sample] += segment * window
    
    # 标准化输出音频
    if np.max(np.abs(output_audio)) > 0:
        output_audio = output_audio / np.max(np.abs(output_audio)) * np.max(np.abs(audio))
    
    return output_audio

# 使用示例
if __name__ == "__main__":
    print(f"项目根目录: {ROOT_DIR}")
    
    # 检查当前工作目录中的音频文件
    print("可用的音频文件:")
    for file in os.listdir(ROOT_DIR):
        if file.endswith(('.mp3', '.wav')):
            print(f"  - {file}")
    
    # 使用绝对路径
    audio_file_name = "暗恋.mp3"
    input_path = os.path.join(ROOT_DIR, audio_file_name)
    
    # 1. 齿音消除处理
    # 加载音频片段（暗恋.mp3 43-49秒）
    start_time = 43
    end_time = 49
    
    try:
        print(f"正在提取音频片段：{input_path} {start_time}-{end_time}秒...")
        audio, sr = extract_audio_segment(input_path, start_time, end_time)
        
        # 生成原始音频的频谱图
        print("生成原始音频频谱图...")
        plot_spectrogram(audio, sr, f"原始音频频谱图 ({audio_file_name}: {start_time}-{end_time}秒)", "原始音频频谱图.png")
        
        # 保存原始音频片段
        print("保存原始音频片段...")
        original_mp3 = "原始音频片段.mp3"
        save_as_mp3(audio, sr, original_mp3)
        
        # 使用加权DPCA算法处理高频聚类
        print("使用加权DPCA算法检测并处理高频聚类...")
        processed_clusters = detect_high_freq_clusters_with_weights(
            audio, sr, 
            time_start=1, time_end=3, 
            freq_threshold=4000, 
            gain_db=-20, 
            n_clusters=3
        )
        
        # 绘制聚类处理后的频谱图
        print("生成聚类处理后的频谱图...")
        plot_spectrogram(processed_clusters, sr, f"高频聚类处理后频谱图 ({audio_file_name}: {start_time}-{end_time}秒)", "聚类处理后频谱图.png")
        
        # 保存聚类处理结果
        print("保存聚类处理后音频...")
        clusters_processed_mp3 = "聚类处理后音频.mp3"
        save_as_mp3(processed_clusters, sr, clusters_processed_mp3)
        
        # 应用齿音消除
        print("应用齿音消除...")
        processed = dynamic_deesser(processed_clusters, sr, 
                                  threshold_db=-18,
                                  reduction_db=10,
                                  crossover=6000)
        
        # 生成处理后音频的频谱图
        print("生成处理后音频频谱图...")
        plot_spectrogram(processed, sr, f"齿音消除后频谱图 ({audio_file_name}: {start_time}-{end_time}秒)", "处理后音频频谱图.png")
        
        # 保存处理结果为mp3格式
        print("保存处理后音频...")
        processed_mp3 = "处理后音频.mp3"
        save_as_mp3(processed, sr, processed_mp3)
        
        print(f"处理完成！\n原始音频：{os.path.join(ROOT_DIR, original_mp3)}\n聚类处理后音频：{os.path.join(ROOT_DIR, clusters_processed_mp3)}\n齿音消除后音频：{os.path.join(ROOT_DIR, processed_mp3)}\n频谱图已保存为PNG文件。")
    except Exception as e:
        print(f"处理齿音消除时出错: {str(e)}")
    
    # 2. 弦乐识别训练与处理
    print("\n开始弦乐处理...")
    string_file_name = "卡拉扬_贝多芬第五交响.mp3"
    string_audio_path = os.path.join(ROOT_DIR, string_file_name)
    
    try:
        if os.path.exists(string_audio_path):
            print("从交响乐中提取弦乐示例...")
            
            # 加载完整音频片段
            full_audio, sr = librosa.load(string_audio_path, sr=None, duration=120)  # 加载前2分钟
            
            # 提取5秒示例片段
            example_start_time = 60  # 秒
            example_duration = 5  # 秒
            example_offset = int(example_start_time * sr)
            example_segment = full_audio[example_offset:example_offset + int(example_duration * sr)]
            
            # 保存原始示例
            string_example_mp3 = "弦乐原始示例_5秒.mp3"
            save_as_mp3(example_segment, sr, string_example_mp3)
            plot_spectrogram(example_segment, sr, "弦乐原始示例频谱图", "弦乐原始示例频谱图.png")
            
            # 使用改进的弦乐提取方法
            print("使用改进的方法提取弦乐部分...")
            extracted_strings = improved_string_extraction(example_segment, sr)
            improved_strings_mp3 = "改进提取的弦乐_5秒.mp3"
            save_as_mp3(extracted_strings, sr, improved_strings_mp3)
            plot_spectrogram(extracted_strings, sr, "改进提取的弦乐频谱图", "改进提取的弦乐频谱图.png")
            
            print(f"弦乐原始示例已保存到: {os.path.join(ROOT_DIR, string_example_mp3)}")
            print(f"提取的弦乐已保存到: {os.path.join(ROOT_DIR, improved_strings_mp3)}")
            
            # 比较两种弦乐提取方法
            if os.path.exists(os.path.join(ROOT_DIR, "string_instrument_classifier.joblib")):
                print("使用机器学习模型提取弦乐以供比较...")
                ml_extracted_strings = extract_string_instruments(example_segment, sr)
                ml_strings_mp3 = "机器学习提取的弦乐_5秒.mp3"
                save_as_mp3(ml_extracted_strings, sr, ml_strings_mp3)
                plot_spectrogram(ml_extracted_strings, sr, "机器学习方法提取的弦乐频谱图", "机器学习提取的弦乐频谱图.png")
                print(f"机器学习提取的弦乐已保存到: {os.path.join(ROOT_DIR, ml_strings_mp3)}")
            else:
                print("原始弦乐识别模型不存在，跳过比较步骤")
                
            # 可视化比较两种方法的频谱
            plt.figure(figsize=(12, 12))
            
            plt.subplot(3, 1, 1)
            S_orig = np.abs(librosa.stft(example_segment))
            librosa.display.specshow(librosa.amplitude_to_db(S_orig, ref=np.max),
                                    y_axis='log', x_axis='time', sr=sr)
            plt.title('原始音频频谱')
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(3, 1, 2)
            S_improved = np.abs(librosa.stft(extracted_strings))
            librosa.display.specshow(librosa.amplitude_to_db(S_improved, ref=np.max),
                                    y_axis='log', x_axis='time', sr=sr)
            plt.title('改进方法提取的弦乐频谱')
            plt.colorbar(format='%+2.0f dB')
            
            if os.path.exists(os.path.join(ROOT_DIR, "string_instrument_classifier.joblib")):
                plt.subplot(3, 1, 3)
                S_ml = np.abs(librosa.stft(ml_extracted_strings))
                librosa.display.specshow(librosa.amplitude_to_db(S_ml, ref=np.max),
                                        y_axis='log', x_axis='time', sr=sr)
                plt.title('机器学习方法提取的弦乐频谱')
                plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(os.path.join(ROOT_DIR, "弦乐提取方法比较.png"))
            plt.close()
            
            print("弦乐提取方法比较图已保存!")
        else:
            print(f"错误: 找不到训练素材文件 '{string_audio_path}'")
            print(f"请将 '{string_file_name}' 文件放到以下目录并再次运行:\n{ROOT_DIR}")
    except Exception as e:
        print(f"弦乐处理时出错: {str(e)}")
        import traceback
        traceback.print_exc()