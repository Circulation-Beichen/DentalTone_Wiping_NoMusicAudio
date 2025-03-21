import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
import json
import pickle
import matplotlib.font_manager as fm
from matplotlib.table import Table
warnings.filterwarnings('ignore')

# 项目根目录路径
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

class SibilanceEnv(gym.Env):
    def __init__(self, audio_path, sr=22050):
        super(SibilanceEnv, self).__init__()
        
        # 加载完整音频
        self.audio, self.sr = librosa.load(audio_path, sr=sr, duration=None)
        
        # 计算梅尔频谱图
        self.mel_spec = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128)
        self.mel_db = librosa.power_to_db(self.mel_spec, ref=np.max)
        
        # 定义动作空间（OPTICS参数）
        self.action_space = gym.spaces.Box(
            low=np.array([0.01, 2, 0.1, 0.01, 0.01]),  # xi, min_samples, min_cluster_size, low_weight, high_weight
            high=np.array([0.1, 10, 0.5, 1.0, 10.0]),
            dtype=np.float32
        )
        
        # 使用固定维度的观察空间
        self.observation_space = gym.spaces.Box(
            low=-80, high=0, shape=(128, 128), dtype=np.float32
        )
        
        # 目标时间点（使用完整6秒音频的时间点）
        self.target_times = [1.6, 2.5, 3.5]
        self.time_tolerance = 0.1
        
        # 计算总时间
        self.total_time = len(self.audio) / self.sr
        print(f"音频总长度: {self.total_time:.2f}秒")
        print(f"梅尔频谱图时间帧数: {self.mel_db.shape[1]}")
        print(f"梅尔频谱图频率数: {self.mel_db.shape[0]}")
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # 处理梅尔频谱图，确保维度一致
        processed_mel_db = self._preprocess_mel_db()
        return processed_mel_db, {}
    
    def _preprocess_mel_db(self):
        """预处理梅尔频谱图，确保维度统一"""
        # 指定统一的时间帧数
        target_frames = 128
        
        # 当前帧数
        current_frames = self.mel_db.shape[1]
        
        # 调整时间维度
        if current_frames > target_frames:
            # 如果当前帧数大于目标帧数，进行降采样
            indices = np.linspace(0, current_frames-1, target_frames, dtype=int)
            processed_mel_db = self.mel_db[:, indices]
        elif current_frames < target_frames:
            # 如果当前帧数小于目标帧数，进行填充
            processed_mel_db = np.zeros((self.mel_db.shape[0], target_frames), dtype=self.mel_db.dtype)
            processed_mel_db[:, :current_frames] = self.mel_db
        else:
            # 如果帧数正好，无需调整
            processed_mel_db = self.mel_db
        
        return processed_mel_db
        
    def step(self, action):
        # 解包动作参数
        xi, min_samples, min_cluster_size, low_weight, high_weight = action
        
        # 提取高频特征
        features = self._extract_features(low_weight, high_weight)
        
        # 使用OPTICS进行聚类
        labels = self._cluster_with_optics(features, xi, min_samples, min_cluster_size, low_weight, high_weight)
        
        # 计算奖励
        reward = self._calculate_reward(labels)
        
        # 判断是否结束
        terminated = True
        truncated = False
        
        # 处理梅尔频谱图，确保维度一致
        processed_mel_db = self._preprocess_mel_db()
        
        return processed_mel_db, reward, terminated, truncated, {}
    
    def _extract_features(self, low_weight, high_weight):
        # 获取时间轴
        times = librosa.frames_to_time(np.arange(self.mel_db.shape[1]), sr=self.sr)
        
        # 获取频率轴
        freqs = librosa.mel_frequencies(n_mels=128)
        
        # 创建特征矩阵
        features = []
        for t in range(self.mel_db.shape[1]):
            # 计算当前时间帧的高频能量
            high_freq_energy = np.mean(self.mel_db[freqs >= 4000, t])
            
            # 计算加权能量
            weighted_energy = 0
            for f in freqs:
                if f < 4000:
                    weight = 0
                elif f < 7000:
                    weight = low_weight
                elif f < 8000:
                    # 在7-8kHz之间平滑过渡
                    ratio = (f - 7000) / 1000
                    weight = low_weight + (high_weight - low_weight) * ratio
                else:
                    weight = high_weight
                weighted_energy += self.mel_db[int(f/self.sr*128), t] * weight
            
            features.append([times[t], high_freq_energy, weighted_energy])
        
        return np.array(features)
    
    def _cluster_with_optics(self, features, xi, min_samples, min_cluster_size, low_weight, high_weight):
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 使用OPTICS进行聚类
        optics = OPTICS(
            xi=float(xi),
            min_samples=int(min_samples),
            min_cluster_size=float(min_cluster_size),
            metric='euclidean'
        )
        labels = optics.fit_predict(features_scaled)
        
        return labels
    
    def _calculate_reward(self, labels):
        # 获取聚类中心
        unique_labels = np.unique(labels[labels != -1])
        cluster_centers = []
        
        for label in unique_labels:
            cluster_points = np.where(labels == label)[0]
            if len(cluster_points) > 0:
                center_time = np.mean(cluster_points) / self.sr
                cluster_centers.append(center_time)
        
        # 计算与目标时间的匹配度
        reward = 0
        for target_time in self.target_times:
            matched = False
            for center_time in cluster_centers:
                if abs(center_time - target_time) <= self.time_tolerance:
                    reward += 1
                    matched = True
                    break
            if not matched:
                reward -= 1
        
        # 根据聚类数量调整奖励
        if len(cluster_centers) == 3:
            reward += 2
        elif len(cluster_centers) > 3:
            reward -= len(cluster_centers) - 3
        else:
            reward -= 3 - len(cluster_centers)
        
        # 添加时间范围奖励
        for center_time in cluster_centers:
            if 0 <= center_time <= self.total_time:
                reward += 0.5
            else:
                reward -= 1
        
        return reward

def train_agent(audio_path, output_dir, continue_training=True, total_timesteps=5000):
    """训练强化学习代理
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        continue_training: 是否继续训练已有模型
        total_timesteps: 训练总步数
    
    Returns:
        训练好的模型
    """
    # 模型保存路径
    model_path = os.path.join(output_dir, "sibilance_agent.zip")
    
    # 创建环境
    env = DummyVecEnv([lambda: SibilanceEnv(audio_path)])
    
    # 检查是否存在已训练的模型
    if os.path.exists(model_path) and continue_training:
        try:
            print(f"找到已有模型，继续训练: {model_path}")
            model = PPO.load(model_path, env=env)
        except ValueError as e:
            print(f"加载模型失败: {e}")
            print("创建新模型开始训练")
            # 删除不兼容的模型
            if os.path.exists(model_path):
                os.rename(model_path, model_path + ".backup")
                print(f"已将不兼容的模型备份为: {model_path}.backup")
            
            # 创建PPO代理
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
    else:
        print("创建新模型开始训练")
        # 创建PPO代理
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
    
    # 训练代理
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return model

def process_audio(audio_path, model, output_dir):
    """使用训练好的模型处理音频"""
    # 创建环境
    env = SibilanceEnv(audio_path)
    
    # 获取最优动作
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    
    # 使用最优参数进行聚类
    xi, min_samples, min_cluster_size, low_weight, high_weight = action
    
    # 提取特征
    features = env._extract_features(low_weight, high_weight)
    
    # 进行聚类
    labels = env._cluster_with_optics(features, xi, min_samples, min_cluster_size, low_weight, high_weight)
    
    # 创建处理后的音频
    processed_audio = env.audio.copy()
    
    # 对每个聚类应用-7dB的负增益
    cluster_info = []
    times_dict = {}  # 用于存储每个聚类的时间点
    
    for label in np.unique(labels[labels != -1]):
        cluster_points = np.where(labels == label)[0]
        if len(cluster_points) > 0:
            # 获取时间点
            times = librosa.frames_to_time(cluster_points, sr=env.sr)
            times_dict[int(label)] = times
            
            # 获取聚类的时间范围
            start_time = times[0]
            end_time = times[-1]
            center_time = np.mean(times)
            
            # 转换为样本索引
            start_sample = int(start_time * env.sr)
            end_sample = int(end_time * env.sr)
            
            # 应用负增益
            gain_factor = 10 ** (-7 / 20)  # -7dB
            processed_audio[start_sample:end_sample] *= gain_factor
            
            # 记录聚类信息
            cluster_info.append({
                'label': int(label),
                'start_time': float(start_time),
                'end_time': float(end_time),
                'center_time': float(center_time),
                'duration': float(end_time - start_time),
                'num_points': len(cluster_points)
            })
    
    # 保存处理后的音频
    output_audio_path = os.path.join(output_dir, "processed_audio.mp3")
    sf.write(output_audio_path, processed_audio, env.sr)
    
    # 绘制并保存带有聚类标记的梅尔图
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(env.mel_db, x_axis='time', y_axis='mel', sr=env.sr)
    plt.colorbar(format='%+2.0f dB')
    
    # 标记聚类
    for label, times in times_dict.items():
        plt.plot(times, [64] * len(times), 'r.', markersize=10, label=f'聚类 {label}')
    
    plt.title('带有聚类标记的梅尔频谱图')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, "mel_spectrogram_with_clusters.png"))
    plt.close()
    
    # 绘制聚类结果表格
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.title('齿音聚类结果')
    
    if cluster_info:
        # 创建表格数据
        col_labels = ['聚类ID', '开始时间(秒)', '结束时间(秒)', '中心时间(秒)', '持续时间(秒)', '点数量']
        table_data = []
        
        for info in cluster_info:
            table_data.append([
                info['label'], 
                f"{info['start_time']:.2f}", 
                f"{info['end_time']:.2f}", 
                f"{info['center_time']:.2f}",
                f"{info['duration']:.2f}",
                info['num_points']
            ])
        
        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.1, 0.15, 0.15, 0.15, 0.15, 0.15]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 为表头设置样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头行
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('darkblue')
            elif j == 0:  # 第一列
                cell.set_text_props(weight='bold')
                cell.set_facecolor('lightgray')
    else:
        plt.text(0.5, 0.5, '未找到聚类', ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 绘制OPTICS参数表格
    plt.figure(figsize=(10, 3))
    plt.axis('off')
    plt.title('OPTICS算法参数')
    
    # 创建参数表格数据
    param_labels = ['xi', 'min_samples', 'min_cluster_size', 'low_weight', 'high_weight']
    param_values = [f"{xi:.3f}", f"{min_samples:.1f}", f"{min_cluster_size:.3f}", f"{low_weight:.3f}", f"{high_weight:.3f}"]
    param_desc = [
        'OPTICS提取层次结构陡度阈值', 
        '核心点最小样本数', 
        '最小聚类大小', 
        '低频权重(4-7kHz)', 
        '高频权重(8kHz以上)'
    ]
    
    param_data = list(zip(param_values, param_desc))
    
    # 创建参数表格
    param_table = plt.table(
        cellText=param_data,
        colLabels=['数值', '描述'],
        rowLabels=param_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.1, 0.6]
    )
    
    # 设置表格样式
    param_table.auto_set_font_size(False)
    param_table.set_fontsize(10)
    param_table.scale(1, 1.5)
    
    # 为表头设置样式
    for (i, j), cell in param_table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('darkblue')
        elif j == -1:  # 行标签列
            cell.set_text_props(weight='bold')
            cell.set_facecolor('lightgray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameters_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存参数和聚类信息
    params = {
        'xi': float(xi),
        'min_samples': float(min_samples),
        'min_cluster_size': float(min_cluster_size),
        'low_weight': float(low_weight),
        'high_weight': float(high_weight),
        'total_time': float(env.total_time),
        'clusters': cluster_info
    }
    
    with open(os.path.join(output_dir, "optimal_parameters.json"), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    return processed_audio, params

def main():
    # 创建输出目录
    output_dir = os.path.join(ROOT_DIR, "OPTICSand强化学习")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 音频文件路径
    audio_path = os.path.join(ROOT_DIR, "原始音频片段_01.mp3")
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件 '{audio_path}' 不存在!")
        audio_files = [f for f in os.listdir(ROOT_DIR) if f.endswith('.mp3')]
        if audio_files:
            print(f"当前目录中的MP3文件: {audio_files}")
            audio_path = os.path.join(ROOT_DIR, audio_files[0])
            print(f"将使用第一个可用文件: {audio_path}")
        else:
            print("目录中没有可用的MP3文件，请确保音频文件存在。")
            return
    
    # 参数配置
    continue_training = True  # 是否继续训练已有模型
    train_new_model = True    # 是否需要训练模型
    total_timesteps = 5000    # 训练总步数
    
    # 模型文件路径
    model_path = os.path.join(output_dir, "sibilance_agent.zip")
    
    if train_new_model:
        # 训练代理
        print("开始训练强化学习代理...")
        model = train_agent(audio_path, output_dir, continue_training, total_timesteps)
    else:
        # 加载已训练的模型
        if os.path.exists(model_path):
            print(f"加载已训练的模型: {model_path}")
            # 创建临时环境用于加载模型
            temp_env = DummyVecEnv([lambda: SibilanceEnv(audio_path)])
            model = PPO.load(model_path, env=temp_env)
        else:
            print(f"未找到已训练的模型: {model_path}，开始训练新模型...")
            model = train_agent(audio_path, output_dir, continue_training=False, total_timesteps=total_timesteps)
    
    # 处理音频
    print("使用训练好的模型处理音频...")
    processed_audio, params = process_audio(audio_path, model, output_dir)
    
    print(f"处理完成！结果保存在: {output_dir}")
    print(f"可以设置 train_new_model = False 直接使用已训练的模型")

if __name__ == "__main__":
    main()
