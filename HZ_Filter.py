import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
import datetime
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
    output_dir = os.path.join(ROOT_DIR, f"HZ滤波实验_阈值滤波_{timestamp}")
    
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

def apply_y_axis_diff_filter(audio, sr, output_dir, threshold_db=-30):
    """在频率轴（y轴）方向应用微分滤波，增加幅度阈值滤波功能"""
    # 计算STFT
    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 转换为dB尺度用于显示和阈值滤波
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 创建阈值掩码
    threshold_mask = magnitude_db < threshold_db
    
    # 准备参数组合
    # 垂直方向的差分滤波器大小（频率方向）
    kernel_heights = [3, 5, 7, 9, 11]  
    # 滤波器强度
    kernel_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]  
    # 差分阶数
    diff_orders = [1, 2]  
    # 是否应用低频截止（布尔值）- 这里我们总是去除低频
    low_freq_cuts = [True]  
    
    # 生成参数组合（目标是80组）
    all_combinations = []
    for height in kernel_heights:
        for intensity in kernel_intensities:
            for order in diff_orders:
                for low_freq_cut in low_freq_cuts:
                    all_combinations.append((height, intensity, order, low_freq_cut))
    
    # 确保至少有80组参数（通过重复或变换其他参数）
    # 由于我们去掉了low_freq_cuts=False的情况，现在只有40组参数
    # 我们可以增加更多的kernel_heights或kernel_intensities组合来补充
    additional_heights = [2, 4, 6, 8, 10, 12]
    additional_intensities = [0.2, 0.4, 0.6, 0.8]
    
    # 添加额外的参数组合
    extra_combinations = []
    for height in additional_heights:
        for intensity in additional_intensities:
            for order in diff_orders:
                extra_combinations.append((height, intensity, order, True))
    
    # 添加额外组合以达到或接近80组
    all_combinations.extend(extra_combinations)
    
    # 确保恰好有80组参数
    while len(all_combinations) > 80:
        all_combinations.pop()
    
    # 计算需要的页数（每页20个子图）
    plots_per_page = 20
    total_pages = (len(all_combinations) + plots_per_page - 1) // plots_per_page
    
    print(f"总参数组合数: {len(all_combinations)}")
    print(f"将生成{total_pages}页图表，每页{plots_per_page}个子图")
    
    # 获取频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-2)
    
    # 创建3kHz以下的频率掩码（用于低频截止）
    low_freq_mask = freqs < 3000
    
    # 保存的最佳参数组合
    best_combinations = [
        (3, 0.3, 1, True),   # 小核，低强度，一阶
        (5, 0.5, 1, True),   # 中核，中强度，一阶
        (7, 0.7, 1, True),   # 大核，高强度，一阶
        (9, 0.5, 2, True),   # 大核，中强度，二阶
        (5, 0.3, 2, True)    # 中核，低强度，二阶
    ]
    
    # 存储处理后的结果
    processed_results = []
    
    # 先绘制原始频谱和阈值滤波后的频谱，进行对比
    plt.figure(figsize=(15, 10))
    
    # 原始频谱
    plt.subplot(2, 1, 1)
    librosa.display.specshow(magnitude_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始频谱图')
    
    # 应用阈值滤波和低频截止
    filtered_magnitude = magnitude.copy()
    filtered_magnitude[threshold_mask] = 0  # 阈值滤波
    filtered_magnitude[low_freq_mask, :] = 0  # 低频截止
    
    # 转换为dB用于显示
    filtered_magnitude_db = librosa.amplitude_to_db(filtered_magnitude, ref=np.max)
    
    # 阈值滤波后的频谱
    plt.subplot(2, 1, 2)
    librosa.display.specshow(filtered_magnitude_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'应用阈值滤波({threshold_db}dB)和低频截止(3kHz)后的频谱图')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "阈值滤波对比图.png"), dpi=300)
    plt.close()
    
    # 按页处理所有参数组合
    for page in range(total_pages):
        plt.figure(figsize=(20, 16))
        
        # 获取当前页的参数组合
        page_start = page * plots_per_page
        page_end = min((page + 1) * plots_per_page, len(all_combinations))
        page_combinations = all_combinations[page_start:page_end]
        
        # 处理当前页的每个参数组合
        for i, (height, intensity, order, low_freq_cut) in enumerate(page_combinations):
            # 从已经应用了阈值滤波和低频截止的幅度谱开始
            filtered_mag = filtered_magnitude.copy()
            
            # 创建垂直方向（频率方向）的差分滤波器
            if order == 1:
                # 一阶差分滤波器
                if height > 1:
                    # 垂直方向的差分卷积核
                    y_diff_kernel = np.ones((height, 1)) / height
                    y_diff_kernel[height//2:, :] *= -1  # 下半部分为负值
            else:
                # 二阶差分滤波器
                y_diff_kernel = np.zeros((height, 1))
                if height >= 3:
                    center = height // 2
                    y_diff_kernel[center, 0] = 2 / height  # 中心为正
                    if center > 0:
                        y_diff_kernel[center-1, 0] = -1 / height  # 上边为负
                    if center < height - 1:
                        y_diff_kernel[center+1, 0] = -1 / height  # 下边为负
            
            # 应用垂直方向（频率轴）的差分滤波
            # 对每一个时间点进行垂直方向的滤波
            filtered_mag_y_diff = filtered_mag.copy()
            
            # 对每一个时间帧应用垂直方向的卷积
            for t in range(filtered_mag.shape[1]):
                # 获取当前时间帧的频率数据（作为列向量）
                freq_column = filtered_mag[:, t:t+1]
                
                # 应用垂直方向的卷积（需要将结果调整为列向量）
                if height > 1 and not np.all(freq_column == 0):  # 避免对全零列进行处理
                    # 使用scipy.signal.convolve2d进行2D卷积
                    from scipy import signal
                    filtered_column = signal.convolve2d(
                        freq_column, y_diff_kernel, mode='same', boundary='symm'
                    )
                    
                    # 混合原始信号和滤波后的信号
                    filtered_mag_y_diff[:, t:t+1] = (
                        freq_column * (1-intensity) + filtered_column * intensity
                    )
            
            # 计算dB值用于显示
            S_db_filtered = librosa.amplitude_to_db(filtered_mag_y_diff, ref=np.max)
            
            # 添加到子图
            plt.subplot(4, 5, i+1)  # 4行5列，每页最多20个子图
            
            librosa.display.specshow(S_db_filtered, x_axis='time', y_axis='log', sr=sr)
            plt.title(f'高度={height},强度={intensity:.1f},阶={order}')
            plt.colorbar(format='%+2.0f dB')
            
            # 生成处理后的音频
            processed_D = filtered_mag_y_diff * np.exp(1j * phase)
            processed_audio = librosa.istft(processed_D)
            
            # 保存处理结果
            result = {
                'height': height,
                'intensity': intensity,
                'order': order,
                'audio': processed_audio,
                'filename': f"HZ滤波_阈值{abs(threshold_db)}dB_h{height}_i{intensity:.1f}_阶{order}.wav"
            }
            processed_results.append(result)
            
            # 保存最佳组合的音频
            if (height, intensity, order, True) in best_combinations:
                save_as_wav(processed_audio, sr, result['filename'], output_dir)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"HZ滤波实验_阈值滤波_Page{page+1}.png"), dpi=300)
        plt.close()
    
    # 创建结果说明文件
    with open(os.path.join(output_dir, "处理结果说明.txt"), "w", encoding="utf-8") as f:
        f.write(f"频率轴微分滤波与阈值滤波实验 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write("处理方法说明：\n")
        f.write(f"本实验使用三步处理方法：\n")
        f.write(f"1. 去除{threshold_db}dB以下的低幅度信号\n")
        f.write(f"2. 去除3kHz以下的低频信号\n")
        f.write(f"3. 在频率轴（y轴）方向应用微分滤波器，增强频谱在垂直方向的变化特征\n\n")
        
        f.write("参数组合:\n")
        f.write(f"- 幅度阈值: {threshold_db}dB\n")
        f.write(f"- 低频截止: 3kHz\n")
        f.write(f"- 滤波器高度: {kernel_heights + additional_heights[:len(all_combinations)-len(kernel_heights)*len(kernel_intensities)*len(diff_orders)]}\n")
        f.write(f"- 滤波器强度: {kernel_intensities + additional_intensities[:4]}\n")
        f.write(f"- 微分阶数: {diff_orders}\n")
        f.write(f"- 总共: {len(all_combinations)}种参数组合\n\n")
        
        f.write("处理结果文件结构:\n")
        f.write(f"- {output_dir}/\n")
        f.write(f"  ├── HZ滤波实验_阈值滤波_Page*.png (实验结果图像)\n")
        f.write(f"  ├── 阈值滤波对比图.png (阈值滤波前后对比)\n")
        f.write(f"  └── 处理结果说明.txt (本文件)\n")
    
    return processed_results

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
    
    # 加载音频文件(指定只加载前30秒)
    start_sec = 0
    end_sec = 30
    y, sr = load_audio(audio_path, start_sec=start_sec, end_sec=end_sec)
    
    # 保存原始音频的频谱图
    plot_spectrogram(y, sr, f'原始音频频谱图: {audio_file} ({start_sec}-{end_sec}秒)', 
                    "原始频谱图.png", output_dir)
    
    # 设置阈值为-30dB，应用阈值滤波、低频截止和频率轴微分滤波
    threshold_db = -30
    processed_results = apply_y_axis_diff_filter(y, sr, audio_dir, threshold_db)
    
    # 创建几个代表性处理后与原始频谱的对比图
    if processed_results:
        plt.figure(figsize=(15, 10))
        
        # 绘制原始频谱
        plt.subplot(2, 3, 1)
        D_orig = librosa.stft(y)
        S_db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
        librosa.display.specshow(S_db_orig, x_axis='time', y_axis='log', sr=sr)
        plt.title('原始频谱图')
        plt.colorbar(format='%+2.0f dB')
        
        # 选择一些代表性的处理结果进行对比
        best_indices = [0, 15, 30, 45, 60]  # 选择几个不同参数的结果
        
        for i, idx in enumerate(best_indices):
            if idx < len(processed_results):
                result = processed_results[idx]
                plt.subplot(2, 3, i+2)
                
                # 计算处理后音频的频谱
                D_proc = librosa.stft(result['audio'])
                S_db_proc = librosa.amplitude_to_db(np.abs(D_proc), ref=np.max)
                
                librosa.display.specshow(S_db_proc, x_axis='time', y_axis='log', sr=sr)
                plt.title(f'h={result["height"]},i={result["intensity"]:.1f},阶={result["order"]}')
                plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "频谱对比图.png"), dpi=300)
        plt.close()
    
    print(f"\n频率轴微分滤波与阈值滤波实验完成! 所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
