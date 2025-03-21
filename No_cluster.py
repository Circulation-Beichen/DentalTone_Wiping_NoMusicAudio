import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
import matplotlib.font_manager as fm

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

def load_audio(audio_path):
    """加载音频文件"""
    print(f"正在加载音频: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"音频加载成功! 采样率: {sr}Hz, 长度: {len(y)/sr:.2f}秒")
    return y, sr

def create_output_folder():
    """创建输出文件夹"""
    output_dir = os.path.join(ROOT_DIR, "预处理无聚类结果")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"创建输出文件夹: {output_dir}")
    return output_dir

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
        processed_audio: 处理后的音频
        phase: 相位信息
        mask: 不为零部分的掩码
        magnitude_db: 原始幅度（dB）
        filtered_magnitude_db: 滤波后的幅度（dB）
    """
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 获取STFT频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-2)
    
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
    
    return filtered_mag, phase, non_zero_mask, magnitude_db, filtered_magnitude_db

def apply_gain_reduction(audio, sr, filtered_mag, phase, non_zero_mask, magnitude_db, threshold_db=-20):
    """
    根据预处理结果对原音频应用负增益
    
    Args:
        audio: 原始音频数据
        sr: 采样率
        filtered_mag: 滤波后的幅度
        phase: 相位信息
        non_zero_mask: 不为零部分的掩码
        magnitude_db: 原始幅度（dB）
        threshold_db: 幅度阈值（dB）
    
    Returns:
        processed_audio: 处理后的音频
        gain_factors: 应用的增益因子
    """
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    
    # 创建增益矩阵（初始化为1，表示不改变）
    gain_factors = np.ones_like(magnitude)
    
    # 获取滤波后不为零的幅度值（dB）
    filtered_magnitude_db = librosa.amplitude_to_db(filtered_mag, ref=np.max)
    
    # 对于不为零的部分，计算负增益
    for f in range(magnitude.shape[0]):
        for t in range(magnitude.shape[1]):
            if non_zero_mask[f, t]:
                x = filtered_magnitude_db[f, t]  # 当前点的dB值
                
                # 确保x确实大于阈值（应该是这样，因为我们已经过滤掉了低于阈值的部分）
                if x > threshold_db:
                    # 计算负增益: (threshold_db-x) dB，这是一个负值（因为x>threshold_db）
                    gain_db = (threshold_db - x)  # 负增益值
                    
                    # 将dB转换为线性增益因子
                    gain_factor = 10 ** (gain_db / 20)
                    
                    # 应用增益因子（值会小于1，因为gain_db是负值）
                    gain_factors[f, t] = gain_factor
                else:
                    # 如果x小于等于阈值（这种情况理论上不应该发生），不应用增益
                    gain_factors[f, t] = 1.0
    
    # 打印一些调试信息以查看增益值范围
    gain_db_values = 20 * np.log10(gain_factors[non_zero_mask])
    if len(gain_db_values) > 0:
        min_gain_db = np.min(gain_db_values)
        max_gain_db = np.max(gain_db_values)
        avg_gain_db = np.mean(gain_db_values)
        print(f"应用的负增益范围: {min_gain_db:.2f}dB 到 {max_gain_db:.2f}dB, 平均: {avg_gain_db:.2f}dB")
    
    # 应用增益
    processed_magnitude = magnitude * gain_factors
    
    # 重建音频
    processed_D = processed_magnitude * np.exp(1j * phase)
    processed_audio = librosa.istft(processed_D)
    
    return processed_audio, gain_factors

def plot_results(audio, processed_audio, sr, magnitude_db, filtered_magnitude_db, gain_factors, output_dir):
    """绘制并保存处理结果"""
    # 创建多个子图
    plt.figure(figsize=(15, 12))
    
    # 绘制原始音频梅尔频谱图
    plt.subplot(3, 1, 1)
    S_orig = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db_orig = librosa.power_to_db(S_orig, ref=np.max)
    librosa.display.specshow(S_db_orig, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始音频梅尔频谱图')
    
    # 绘制处理后的音频梅尔频谱图
    plt.subplot(3, 1, 2)
    S_proc = librosa.feature.melspectrogram(y=processed_audio, sr=sr)
    S_db_proc = librosa.power_to_db(S_proc, ref=np.max)
    librosa.display.specshow(S_db_proc, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('处理后音频梅尔频谱图')
    
    # 绘制应用的增益因子（dB）
    plt.subplot(3, 1, 3)
    gain_db = librosa.amplitude_to_db(gain_factors, ref=np.max)
    librosa.display.specshow(gain_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('应用的负增益（dB）')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "处理结果对比.png"), dpi=300)
    plt.close()
    
    # 单独绘制滤波器处理前后的频谱图对比
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(magnitude_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('滤波器处理前的频谱图')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(filtered_magnitude_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'滤波器处理后的频谱图 (阈值={threshold_db}dB, w={width}, i={intensity:.1f}, 阶={order})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "滤波器处理对比.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # 指定参数
    threshold_db = -20
    width = 7
    intensity = 0.7
    order = 1
    
    print(f"使用参数: 阈值={threshold_db}dB, 滤波器宽度={width}, 强度={intensity}, 阶数={order}")
    
    # 指定音频文件
    audio_file = "原始音频片段_01.mp3"
    audio_path = os.path.join(ROOT_DIR, audio_file)
    
    # 创建输出文件夹
    output_dir = create_output_folder()
    
    # 加载音频
    audio, sr = load_audio(audio_path)
    
    # 预处理音频
    print("预处理音频...")
    filtered_mag, phase, non_zero_mask, magnitude_db, filtered_magnitude_db = preprocess_audio(
        audio, sr, threshold_db, width, intensity, order
    )
    
    # 应用负增益
    print("应用负增益...")
    processed_audio, gain_factors = apply_gain_reduction(
        audio, sr, filtered_mag, phase, non_zero_mask, magnitude_db, threshold_db
    )
    
    # 保存处理后的音频
    output_audio_path = os.path.join(output_dir, "处理后音频.wav")
    sf.write(output_audio_path, processed_audio, sr)
    print(f"处理后音频已保存至: {output_audio_path}")
    
    # 绘制结果
    print("生成可视化结果...")
    plot_results(audio, processed_audio, sr, magnitude_db, filtered_magnitude_db, gain_factors, output_dir)
    
    # 创建处理说明
    with open(os.path.join(output_dir, "处理说明.txt"), 'w', encoding='utf-8') as f:
        f.write("无聚类预处理处理说明\n")
        f.write("="*30 + "\n\n")
        f.write(f"处理参数:\n")
        f.write(f"- 音频文件: {audio_file}\n")
        f.write(f"- 幅度阈值: {threshold_db} dB\n")
        f.write(f"- 滤波器宽度: {width}\n")
        f.write(f"- 滤波器强度: {intensity}\n")
        f.write(f"- 微分阶数: {order}\n\n")
        f.write("处理步骤:\n")
        f.write("1. 删除3kHz以下频率\n")
        f.write("2. 删除低于阈值的幅度\n")
        f.write("3. 应用微分滤波器\n")
        f.write("4. 对非零部分应用负增益: 如果某点的幅度为x dB且x大于阈值，则应用(阈值-x) dB的负增益\n\n")
        f.write("处理结果:\n")
        f.write(f"- 输出音频: {output_audio_path}\n")
        f.write("- 可视化图表展示了原始音频频谱、处理后音频频谱以及应用的增益\n")
    
    print(f"处理完成! 结果保存在: {output_dir}")
