import numpy as np
import librosa
import soundfile as sf
import os
import joblib
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 项目根目录路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def plot_spectrogram(y, sr, title, filename, output_dir=None):
    """生成并保存频谱图"""
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "处理结果", "频谱图")
    
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"频谱图已保存到: {output_path}")
    plt.close()

def train_string_instrument_classifier(audio_path, segment_duration=3, overlap=1.5, output_dir=None):
    """
    使用卡拉扬_贝多芬第五交响乐的音频训练弦乐识别模型
    :param audio_path: 音频文件路径
    :param segment_duration: 每个训练样本的持续时间(秒)
    :param overlap: 相邻样本之间的重叠时间(秒)
    :param output_dir: 输出目录
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "处理结果", "弦乐识别模型")
    
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    model_path = os.path.join(output_dir, "string_instrument_classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 创建模型评估结果文件
    with open(os.path.join(output_dir, "模型评估结果.txt"), "w", encoding="utf-8") as f:
        f.write(f"模型准确率: {accuracy:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(report)
    
    # 绘制一些弦乐和非弦乐片段的频谱图作为参考
    examples_dir = os.path.join(output_dir, "示例频谱图")
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    
    for i, label_name in enumerate(['非弦乐主导', '弦乐主导']):
        # 找到对应标签的前两个样本
        indices = [idx for idx, label in enumerate(labels) if label == i][:2]
        
        for j, idx in enumerate(indices):
            # 绘制频谱图
            segment = segments[idx]
            plot_spectrogram(segment, sr, f"{label_name}频谱示例 #{j+1}", f"{label_name}_示例_{j+1}.png", examples_dir)
    
    return clf, accuracy

def improved_string_extraction(audio, sr, output_dir=None):
    """
    改进的弦乐提取方法，使用频谱特征和谐波跟踪
    :param audio: 输入音频
    :param sr: 采样率
    :param output_dir: 输出目录
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

def extract_string_instruments(audio, sr, model_path=None, output_dir=None):
    """
    从音频中提取弦乐部分
    :param audio: 输入音频信号
    :param sr: 采样率
    :param model_path: 训练好的弦乐识别模型路径
    :param output_dir: 输出目录
    :return: 仅包含弦乐的音频信号
    """
    # 设置默认模型路径
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "处理结果", "弦乐识别模型")
    
    if model_path is None:
        model_path = os.path.join(output_dir, "string_instrument_classifier.joblib")
    
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
