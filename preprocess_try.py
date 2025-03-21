import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
from scipy import signal
import datetime
import shutil
import matplotlib.font_manager as fm
import itertools

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
    output_dir = os.path.join(ROOT_DIR, f"预处理实验_{timestamp}")
    
    # 创建主输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建子文件夹
    method1_dir = os.path.join(output_dir, "方法1_删除低频_微分滤波")
    method2_dir = os.path.join(output_dir, "方法2_删除低幅度")
    method3_dir = os.path.join(output_dir, "方法3_删除低幅度_微分滤波")
    audio_dir = os.path.join(output_dir, "处理后音频")
    
    for directory in [method1_dir, method2_dir, method3_dir, audio_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    print(f"创建输出文件夹: {output_dir}")
    return output_dir, method1_dir, method2_dir, method3_dir, audio_dir

def load_audio(audio_path, start_sec=None, end_sec=None):
    """加载音频文件"""
    print(f"正在加载音频: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    
    # 如果没有指定时间范围，默认只加载前30秒
    if start_sec is None:
        start_sec = 0
    if end_sec is None:
        end_sec = 30  # 默认只加载30秒音频，避免内存溢出
    
    print(f"加载音频片段: {start_sec}-{end_sec}秒")
    
    # 加载音频片段
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

def method1_remove_lowfreq_diff_filter(audio, sr, output_dir):
    """方法1: 删除3kHz以下频率 + 微分滤波"""
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 获取STFT频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-2)
    
    # 创建3kHz以下的频率掩码
    low_freq_mask = freqs < 3000
    
    # 删除3kHz以下的频率
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[low_freq_mask, :] = 0
    
    # 准备不同大小的差分滤波器
    kernel_widths = [3, 5, 7, 9]  # 滤波器宽度
    kernel_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]  # 滤波器强度
    
    # 生成所有参数组合的笛卡尔积
    combinations = list(itertools.product(kernel_widths, kernel_intensities))
    
    # 准备存储处理结果
    processed_magnitudes = []
    processed_audios = []
    
    # 每张图最多20个子图，计算需要的图数量
    num_plots = len(combinations)
    plots_per_page = 20
    num_pages = (num_plots + plots_per_page - 1) // plots_per_page
    
    print(f"方法1: 将生成{num_pages}页图表, 每页最多{plots_per_page}个子图")
    
    for page in range(num_pages):
        plt.figure(figsize=(20, 16))
        page_combinations = combinations[page*plots_per_page:(page+1)*plots_per_page]
        
        for i, (width, intensity) in enumerate(page_combinations):
            # 创建差分滤波器
            diff_kernel = np.ones(width) / width
            diff_kernel[width//2:] *= -1  # 将后半部分设为负值，创建差分效果
            
            # 应用差分滤波
            filtered_mag = filtered_magnitude.copy()
            for f_idx in range(filtered_mag.shape[0]):
                # 仅对已保留的频率应用差分滤波（3kHz以上）
                if not low_freq_mask[f_idx]:
                    # 应用卷积
                    filtered_row = np.convolve(filtered_mag[f_idx, :], diff_kernel, mode='same')
                    # 应用强度参数
                    filtered_mag[f_idx, :] = filtered_mag[f_idx, :] * (1-intensity) + filtered_row * intensity
            
            # 反变换回时域
            processed_D = filtered_mag * np.exp(1j * phase)
            processed_audio = librosa.istft(processed_D)
            processed_audios.append((processed_audio, f"方法1_w{width}_i{intensity:.1f}.wav"))
            
            # 为频谱图计算dB值
            S_db = librosa.amplitude_to_db(filtered_mag, ref=np.max)
            
            # 添加到子图
            plt.subplot(4 if len(page_combinations) > 16 else 5, 
                        5 if len(page_combinations) > 16 else 4, 
                        i+1)
            
            librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
            plt.title(f'滤波器宽度={width}, 强度={intensity:.1f}')
            plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"方法1_微分滤波_Page{page+1}.png"), dpi=300)
        plt.close()
    
    # 保存处理后的音频
    best_combinations = [(3, 0.5), (5, 0.5), (7, 0.3), (9, 0.1)]  # 保存一些代表性的组合
    for audio, filename in processed_audios:
        # 检查是否为最佳组合之一
        width = int(filename.split('_w')[1].split('_')[0])
        intensity = float(filename.split('_i')[1].split('.wav')[0])
        if (width, intensity) in best_combinations:
            save_as_wav(audio, sr, filename, output_dir)
    
    return processed_audios

def method2_remove_lowdb(audio, sr, output_dir, threshold_db=-30):
    """方法2: 删除-30dB以下幅度"""
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 转换为dB尺度
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 创建阈值掩码
    mask = magnitude_db < threshold_db
    
    # 应用掩码，删除低幅度成分
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[mask] = 0
    
    # 绘制处理前后的频谱图对比
    plt.figure(figsize=(14, 8))
    
    # 原始频谱图
    plt.subplot(2, 1, 1)
    librosa.display.specshow(magnitude_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始频谱图')
    
    # 处理后频谱图
    plt.subplot(2, 1, 2)
    S_db_filtered = librosa.amplitude_to_db(filtered_magnitude, ref=np.max)
    librosa.display.specshow(S_db_filtered, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'删除{threshold_db}dB以下幅度后')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"方法2_删除{abs(threshold_db)}dB以下幅度.png"), dpi=300)
    plt.close()
    
    # 反变换回时域
    processed_D = filtered_magnitude * np.exp(1j * phase)
    processed_audio = librosa.istft(processed_D)
    
    # 保存处理后的音频
    filename = f"方法2_删除{abs(threshold_db)}dB以下.wav"
    save_as_wav(processed_audio, sr, filename, output_dir)
    
    return processed_audio

def method3_remove_lowdb_diff_filter(audio, sr, output_dir, threshold_db=-30):
    """方法3: 删除-30dB以下幅度 + 微分滤波"""
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 转换为dB尺度
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 创建阈值掩码，删除低幅度成分
    mask = magnitude_db < threshold_db
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[mask] = 0
    
    # 准备不同大小的差分滤波器
    kernel_widths = [3, 5, 7, 9]  # 滤波器宽度
    kernel_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]  # 滤波器强度
    
    # 生成所有参数组合的笛卡尔积
    combinations = list(itertools.product(kernel_widths, kernel_intensities))
    
    # 准备存储处理结果
    processed_magnitudes = []
    processed_audios = []
    
    # 每张图最多20个子图，计算需要的图数量
    num_plots = len(combinations)
    plots_per_page = 20
    num_pages = (num_plots + plots_per_page - 1) // plots_per_page
    
    print(f"方法3: 将生成{num_pages}页图表, 每页最多{plots_per_page}个子图")
    
    for page in range(num_pages):
        plt.figure(figsize=(20, 16))
        page_combinations = combinations[page*plots_per_page:(page+1)*plots_per_page]
        
        for i, (width, intensity) in enumerate(page_combinations):
            # 创建差分滤波器
            diff_kernel = np.ones(width) / width
            diff_kernel[width//2:] *= -1  # 将后半部分设为负值，创建差分效果
            
            # 应用差分滤波
            filtered_mag = filtered_magnitude.copy()
            for f_idx in range(filtered_mag.shape[0]):
                # 对非零幅度的频率应用差分滤波
                if not np.all(filtered_mag[f_idx, :] == 0):
                    # 应用卷积
                    filtered_row = np.convolve(filtered_mag[f_idx, :], diff_kernel, mode='same')
                    # 应用强度参数
                    filtered_mag[f_idx, :] = filtered_mag[f_idx, :] * (1-intensity) + filtered_row * intensity
            
            # 反变换回时域
            processed_D = filtered_mag * np.exp(1j * phase)
            processed_audio = librosa.istft(processed_D)
            processed_audios.append((processed_audio, f"方法3_w{width}_i{intensity:.1f}.wav"))
            
            # 为频谱图计算dB值
            S_db = librosa.amplitude_to_db(filtered_mag, ref=np.max)
            
            # 添加到子图
            plt.subplot(4 if len(page_combinations) > 16 else 5, 
                        5 if len(page_combinations) > 16 else 4, 
                        i+1)
            
            librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
            plt.title(f'滤波器宽度={width}, 强度={intensity:.1f}')
            plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"方法3_删除低幅度_微分滤波_Page{page+1}.png"), dpi=300)
        plt.close()
    
    # 保存处理后的音频
    best_combinations = [(3, 0.5), (5, 0.5), (7, 0.3), (9, 0.1)]  # 保存一些代表性的组合
    for audio, filename in processed_audios:
        # 检查是否为最佳组合之一
        width = int(filename.split('_w')[1].split('_')[0])
        intensity = float(filename.split('_i')[1].split('.wav')[0])
        if (width, intensity) in best_combinations:
            save_as_wav(audio, sr, filename, output_dir)
    
    return processed_audios

def main():
    # 创建输出文件夹
    output_dir, method1_dir, method2_dir, method3_dir, audio_dir = create_output_folders()
    
    # 指定使用的音频文件
    audio_file = "原始音频片段_01.mp3"
    audio_path = os.path.join(ROOT_DIR, audio_file)
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件 '{audio_path}' 不存在!")
        audio_files = [f for f in os.listdir(ROOT_DIR) if f.endswith('.mp3')]
        if audio_files:
            print(f"当前目录中的MP3文件: {audio_files}")
            print("请确保'原始音频片段_01.mp3'文件存在")
        else:
            print("目录中没有MP3文件，请添加所需的音频文件")
        return
    
    print(f"将使用音频文件: {audio_file}")
    
    # 加载音频文件(指定只加载前30秒)
    start_sec = 0
    end_sec = 30
    y, sr = load_audio(audio_path, start_sec=start_sec, end_sec=end_sec)
    
    # 保存原始音频的频谱图
    plot_spectrogram(y, sr, f'原始音频频谱图: {audio_file} ({start_sec}-{end_sec}秒)', "原始频谱图.png", output_dir)
    
    # 预处理实验参数设置 - 总共80组参数组合
    threshold_dbs = [-20, -30, -35]  # 幅度阈值，三个级别
    kernel_widths = [3, 5, 7, 9, 11]  # 滤波器宽度，五个级别
    kernel_intensities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # 滤波器强度，六个级别
    diff_orders = [1, 2]  # 微分阶数，两个级别
    
    # 计算每个阈值的参数组合数
    combinations_per_threshold = len(kernel_widths) * len(kernel_intensities) * len(diff_orders)
    total_combinations = combinations_per_threshold * len(threshold_dbs)
    
    print(f"总参数组合数: {total_combinations}，每个阈值有{combinations_per_threshold}个组合")
    
    # 设置每页最多显示的子图数量
    plots_per_page = 20
    total_pages = (total_combinations + plots_per_page - 1) // plots_per_page
    
    print(f"将生成{total_pages}页图表，每页最多{plots_per_page}个子图")
    
    # 生成所有参数组合
    all_combinations = []
    for threshold_db in threshold_dbs:
        for width in kernel_widths:
            for intensity in kernel_intensities:
                for order in diff_orders:
                    all_combinations.append((threshold_db, width, intensity, order))
    
    # 开始批量处理
    processed_results = []
    
    # 按页处理所有组合
    for page in range(total_pages):
        plt.figure(figsize=(20, 16))
        
        # 获取当前页的参数组合
        page_start = page * plots_per_page
        page_end = min((page + 1) * plots_per_page, len(all_combinations))
        page_combinations = all_combinations[page_start:page_end]
        
        # 处理当前页的每个参数组合
        for i, (threshold_db, width, intensity, order) in enumerate(page_combinations):
            # 计算STFT
            D = librosa.stft(y)
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
            # 创建差分滤波器 (不同阶数)
            if order == 1:
                # 一阶微分滤波器
                diff_kernel = np.ones(width) / width
                diff_kernel[width//2:] *= -1  # 将后半部分设为负值，创建差分效果
            else:
                # 二阶微分滤波器 (更强的边缘检测)
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
            
            # 为频谱图计算dB值
            S_db = librosa.amplitude_to_db(filtered_mag, ref=np.max)
            
            # 添加到子图
            plt.subplot(4, 5, i+1)  # 4行5列，最多20个子图
            
            librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
            plt.title(f'阈值={threshold_db}dB,w={width},i={intensity:.1f},阶={order}')
            plt.colorbar(format='%+2.0f dB')
            
            # 保存处理结果信息
            processed_results.append({
                'threshold_db': threshold_db,
                'width': width,
                'intensity': intensity,
                'order': order
            })
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"预处理实验_Page{page+1}.png"), dpi=300)
        plt.close()
    
    # 创建结果说明文件
    with open(os.path.join(output_dir, "处理结果说明.txt"), "w", encoding="utf-8") as f:
        f.write(f"音频预处理实验 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write("处理方法说明：\n")
        f.write("对音频进行三步处理:\n")
        f.write("1. 删除3kHz以下频率\n")
        f.write("2. 删除低于阈值的幅度\n")
        f.write("3. 应用微分滤波器\n\n")
        
        f.write("参数组合:\n")
        f.write(f"- 幅度阈值: {threshold_dbs} dB, 共{len(threshold_dbs)}个级别\n")
        f.write(f"- 滤波器宽度: {kernel_widths}, 共{len(kernel_widths)}个级别\n")
        f.write(f"- 滤波器强度: {kernel_intensities}, 共{len(kernel_intensities)}个级别\n")
        f.write(f"- 微分阶数: {diff_orders}, 共{len(diff_orders)}个级别\n")
        f.write(f"- 总共: {total_combinations}种参数组合\n\n")
        
        f.write("处理结果文件结构:\n")
        f.write(f"- {output_dir}/\n")
        f.write(f"  ├── 预处理实验_Page*.png (实验结果图像)\n")
        f.write(f"  └── 处理结果说明.txt (本文件)\n")
    
    print(f"\n预处理实验完成! 所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
