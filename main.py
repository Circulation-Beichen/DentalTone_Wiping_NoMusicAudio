import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
import warnings
import shutil
import datetime
warnings.filterwarnings('ignore')

# 导入模块
from recognition import (
    plot_spectrogram, improved_string_extraction, extract_string_instruments,
    train_string_instrument_classifier
)
from OPTICS import (
    dynamic_deesser, detect_high_freq_clusters_with_optics
)

# 项目根目录路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_output_folders():
    """创建输出文件夹结构"""
    # 以日期时间命名主输出文件夹
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(ROOT_DIR, f"处理结果_{timestamp}")
    
    # 创建主输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建子文件夹
    audio_dir = os.path.join(output_dir, "音频文件")
    spectrogram_dir = os.path.join(output_dir, "频谱图")
    optics_dir = os.path.join(output_dir, "OPTICS分析")
    models_dir = os.path.join(output_dir, "弦乐识别模型")
    
    for directory in [audio_dir, spectrogram_dir, optics_dir, models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    print(f"创建输出文件夹: {output_dir}")
    return output_dir, audio_dir, spectrogram_dir, optics_dir, models_dir

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

def save_as_mp3(audio, sr, filename, output_dir):
    """保存为mp3格式"""
    # 确保使用绝对路径
    output_path = os.path.join(output_dir, filename)
    
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
    
    return output_path

# 使用示例
if __name__ == "__main__":
    print(f"项目根目录: {ROOT_DIR}")
    
    # 检查当前工作目录中的音频文件
    print("可用的音频文件:")
    for file in os.listdir(ROOT_DIR):
        if file.endswith(('.mp3', '.wav')):
            print(f"  - {file}")
    
    # 创建输出文件夹结构
    output_dir, audio_dir, spectrogram_dir, optics_dir, models_dir = create_output_folders()
    
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
        plot_spectrogram(audio, sr, f"原始音频频谱图 ({audio_file_name}: {start_time}-{end_time}秒)", "原始音频频谱图.png", spectrogram_dir)
        
        # 保存原始音频片段
        print("保存原始音频片段...")
        original_mp3 = "原始音频片段.mp3"
        original_mp3_path = save_as_mp3(audio, sr, original_mp3, audio_dir)
        
        # 使用OPTICS算法处理高频聚类
        print("使用OPTICS算法检测并处理高频聚类...")
        processed_clusters = detect_high_freq_clusters_with_optics(
            audio, sr, 
            time_start=1, time_end=3, 
            freq_threshold=4000, 
            max_reduction_db=10, 
            min_samples=5,
            xi=0.05,
            min_cluster_size=0.05,
            output_dir=output_dir
        )
        
        # 绘制聚类处理后的频谱图
        print("生成聚类处理后的频谱图...")
        plot_spectrogram(processed_clusters, sr, f"高频聚类处理后频谱图 ({audio_file_name}: {start_time}-{end_time}秒)", "聚类处理后频谱图.png", spectrogram_dir)
        
        # 保存聚类处理结果
        print("保存聚类处理后音频...")
        clusters_processed_mp3 = "聚类处理后音频.mp3"
        clusters_processed_mp3_path = save_as_mp3(processed_clusters, sr, clusters_processed_mp3, audio_dir)
        
        # 应用齿音消除
        print("应用齿音消除...")
        processed = dynamic_deesser(processed_clusters, sr, 
                                  threshold_db=-18,
                                  reduction_db=10,
                                  crossover=6000)
        
        # 生成处理后音频的频谱图
        print("生成处理后音频频谱图...")
        plot_spectrogram(processed, sr, f"齿音消除后频谱图 ({audio_file_name}: {start_time}-{end_time}秒)", "处理后音频频谱图.png", spectrogram_dir)
        
        # 保存处理结果为mp3格式
        print("保存处理后音频...")
        processed_mp3 = "处理后音频.mp3"
        processed_mp3_path = save_as_mp3(processed, sr, processed_mp3, audio_dir)
        
        print(f"处理完成！\n原始音频：{original_mp3_path}\n聚类处理后音频：{clusters_processed_mp3_path}\n齿音消除后音频：{processed_mp3_path}\n频谱图已保存至：{spectrogram_dir}")
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
            
            # 创建弦乐处理的子文件夹
            string_audio_dir = os.path.join(audio_dir, "弦乐处理")
            if not os.path.exists(string_audio_dir):
                os.makedirs(string_audio_dir)
            
            string_spectrograms_dir = os.path.join(spectrogram_dir, "弦乐处理")
            if not os.path.exists(string_spectrograms_dir):
                os.makedirs(string_spectrograms_dir)
            
            # 保存原始示例
            string_example_mp3 = "弦乐原始示例_5秒.mp3"
            string_example_mp3_path = save_as_mp3(example_segment, sr, string_example_mp3, string_audio_dir)
            plot_spectrogram(example_segment, sr, "弦乐原始示例频谱图", "弦乐原始示例频谱图.png", string_spectrograms_dir)
            
            # 使用改进的弦乐提取方法
            print("使用改进的方法提取弦乐部分...")
            extracted_strings = improved_string_extraction(example_segment, sr, output_dir)
            improved_strings_mp3 = "改进提取的弦乐_5秒.mp3"
            improved_strings_mp3_path = save_as_mp3(extracted_strings, sr, improved_strings_mp3, string_audio_dir)
            plot_spectrogram(extracted_strings, sr, "改进提取的弦乐频谱图", "改进提取的弦乐频谱图.png", string_spectrograms_dir)
            
            print(f"弦乐原始示例已保存到: {string_example_mp3_path}")
            print(f"提取的弦乐已保存到: {improved_strings_mp3_path}")
            
            # 训练模型
            print("训练弦乐识别模型...")
            train_string_instrument_classifier(string_audio_path, output_dir=models_dir)
            
            # 比较两种弦乐提取方法
            import os.path
            model_path = os.path.join(models_dir, "string_instrument_classifier.joblib")
            if os.path.exists(model_path):
                print("使用机器学习模型提取弦乐以供比较...")
                ml_extracted_strings = extract_string_instruments(example_segment, sr, model_path, models_dir)
                ml_strings_mp3 = "机器学习提取的弦乐_5秒.mp3"
                ml_strings_mp3_path = save_as_mp3(ml_extracted_strings, sr, ml_strings_mp3, string_audio_dir)
                plot_spectrogram(ml_extracted_strings, sr, "机器学习方法提取的弦乐频谱图", "机器学习提取的弦乐频谱图.png", string_spectrograms_dir)
                print(f"机器学习提取的弦乐已保存到: {ml_strings_mp3_path}")
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
            
            if os.path.exists(model_path):
                plt.subplot(3, 1, 3)
                S_ml = np.abs(librosa.stft(ml_extracted_strings))
                librosa.display.specshow(librosa.amplitude_to_db(S_ml, ref=np.max),
                                        y_axis='log', x_axis='time', sr=sr)
                plt.title('机器学习方法提取的弦乐频谱')
                plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(os.path.join(string_spectrograms_dir, "弦乐提取方法比较.png"))
            plt.close()
            
            print("弦乐提取方法比较图已保存!")
            
            # 创建处理结果说明文件
            with open(os.path.join(output_dir, "处理结果说明.txt"), "w", encoding="utf-8") as f:
                f.write(f"音频处理结果 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                f.write("1. 齿音消除处理\n")
                f.write(f"   - 原始音频: {os.path.basename(original_mp3_path)}\n")
                f.write(f"   - 聚类处理后音频: {os.path.basename(clusters_processed_mp3_path)}\n")
                f.write(f"   - 齿音消除后音频: {os.path.basename(processed_mp3_path)}\n\n")
                f.write("2. 弦乐处理\n")
                f.write(f"   - 原始弦乐示例: {os.path.basename(string_example_mp3_path)}\n")
                f.write(f"   - 改进方法提取的弦乐: {os.path.basename(improved_strings_mp3_path)}\n")
                if os.path.exists(model_path):
                    f.write(f"   - 机器学习提取的弦乐: {os.path.basename(ml_strings_mp3_path)}\n")
                f.write("\n")
                f.write("详细频谱图和分析图表可在相应文件夹中查看。\n")
            
            print(f"\n所有处理结果已保存到文件夹: {output_dir}")
            
        else:
            print(f"错误: 找不到训练素材文件 '{string_audio_path}'")
            print(f"请将 '{string_file_name}' 文件放到以下目录并再次运行:\n{ROOT_DIR}")
    except Exception as e:
        print(f"弦乐处理时出错: {str(e)}")
        import traceback
        traceback.print_exc()