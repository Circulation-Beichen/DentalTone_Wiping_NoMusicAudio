import os
import sys
import datetime
import threading
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QProgressBar, QFileDialog, QFrame, QSlider)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入HZ_Filter模块中的函数
from HZ_Filter import setup_chinese_font, process_audio_file

# 设置matplotlib中文字体
setup_chinese_font()

class AudioProcessingThread(QThread):
    """音频处理线程，避免界面阻塞"""
    progress_signal = pyqtSignal(int)  # 进度信号
    result_signal = pyqtSignal(str, str)  # 结果信号 (原始音频路径, 处理后音频路径)
    finished_signal = pyqtSignal(str, str, str)  # 处理完成信号 (输出目录, 原始梅尔图路径, 处理后梅尔图路径)
    error_signal = pyqtSignal(str)  # 错误信号

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path
        
    def run(self):
        try:
            # 创建输出目录
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     f"{timestamp}_嘿，结果在这里")
            
            self.progress_signal.emit(10)
            
            # 处理音频文件
            processed_audio, sr, output_dir = process_audio_file(
                self.audio_path,
                output_dir=output_dir,
                target_sr=16000,
                threshold_db=-35,
                max_negative_gain_db=-20,
                height=10,
                intensity=0.5,
                order=2,
                save_intermediate=True,
                plot_spectrograms=True
            )
            
            self.progress_signal.emit(90)
            
            # 获取处理后音频文件路径
            processed_file = os.path.join(output_dir, "处理后音频", 
                                         "最终处理_负增益-20dB_h10_i0.5_阶2.wav")
            
            # 获取梅尔频谱图路径
            original_mel_spec = os.path.join(output_dir, "原始梅尔频谱图.png")
            processed_mel_spec = os.path.join(output_dir, "处理后梅尔频谱图.png")
            
            # 发送结果信号
            self.result_signal.emit(self.audio_path, processed_file)
            self.finished_signal.emit(output_dir, original_mel_spec, processed_mel_spec)
            
            self.progress_signal.emit(100)
            
        except Exception as e:
            self.error_signal.emit(f"处理过程中出错: {str(e)}")


class DropAreaWidget(QWidget):
    """可拖放文件的区域"""
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        
        # 设置样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px dashed #aaaaaa;
                border-radius: 5px;
            }
        """)
        
        # 添加提示标签
        layout = QVBoxLayout()
        
        self.label = QLabel("将MP3文件拖放到这里\n或点击选择文件")
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        
        layout.addWidget(self.label)
        self.setLayout(layout)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        # 检查是否是文件
        if event.mimeData().hasUrls():
            # 检查是否是mp3文件
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith('.mp3'):
                event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        # 获取文件路径
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.file_dropped.emit(file_path)
        

class AudioPlayerWidget(QWidget):
    """音频播放器控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout()
        
        # 创建标签
        self.title_label = QLabel("音频播放器")
        self.title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.title_label.setFont(font)
        
        # 创建播放控制按钮
        self.buttons_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放原始")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_original)
        
        self.play_processed_button = QPushButton("播放处理后")
        self.play_processed_button.setEnabled(False)
        self.play_processed_button.clicked.connect(self.play_processed)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_playback)
        
        self.buttons_layout.addWidget(self.play_button)
        self.buttons_layout.addWidget(self.play_processed_button)
        self.buttons_layout.addWidget(self.stop_button)
        
        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self.set_position)
        
        # 将控件添加到布局
        self.layout.addWidget(self.title_label)
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.progress_slider)
        
        self.setLayout(self.layout)
        
        # 创建媒体播放器
        self.player = QMediaPlayer()
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.stateChanged.connect(self.state_changed)
        
        # 音频文件路径
        self.original_audio = None
        self.processed_audio = None
        
    def set_audio_files(self, original_audio, processed_audio):
        """设置音频文件路径"""
        self.original_audio = original_audio
        self.processed_audio = processed_audio
        self.play_button.setEnabled(True)
        self.play_processed_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.progress_slider.setEnabled(True)
    
    def play_original(self):
        """播放原始音频"""
        if self.original_audio:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.original_audio)))
            self.player.play()
            self.title_label.setText("正在播放: 原始音频")
    
    def play_processed(self):
        """播放处理后音频"""
        if self.processed_audio:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.processed_audio)))
            self.player.play()
            self.title_label.setText("正在播放: 处理后音频")
    
    def stop_playback(self):
        """停止播放"""
        self.player.stop()
        self.title_label.setText("音频播放器")
    
    def position_changed(self, position):
        """播放位置变化"""
        self.progress_slider.setValue(position)
    
    def duration_changed(self, duration):
        """音频时长变化"""
        self.progress_slider.setRange(0, duration)
    
    def state_changed(self, state):
        """播放状态变化"""
        if state == QMediaPlayer.StoppedState:
            self.title_label.setText("音频播放器")
    
    def set_position(self, position):
        """设置播放位置"""
        self.player.setPosition(position)


class SpectrogramWidget(QWidget):
    """频谱图显示控件"""
    def __init__(self, title="频谱图", parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout()
        
        # 创建标签
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.title_label.setFont(font)
        
        # 创建图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 180)
        self.image_label.setStyleSheet("background-color: white;")
        
        # 添加到布局
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)
        
        self.setLayout(self.layout)
    
    def set_image(self, image_path):
        """设置频谱图图像"""
        if image_path and os.path.exists(image_path):
            # 加载图像并保持纵横比缩放
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        else:
            self.image_label.setText("图像不可用")


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        
        # 音频文件路径
        self.current_audio = None
        self.processed_audio = None
        self.output_dir = None
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("音频去齿音处理")
        self.setMinimumSize(800, 600)
        
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 顶部控件：文件拖放区
        self.drop_area = DropAreaWidget()
        self.drop_area.file_dropped.connect(self.process_audio_file)
        main_layout.addWidget(self.drop_area)
        
        # 点击选择文件按钮
        self.select_file_button = QPushButton("选择MP3文件")
        self.select_file_button.clicked.connect(self.select_audio_file)
        main_layout.addWidget(self.select_file_button)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 频谱图显示区域
        spectrograms_layout = QHBoxLayout()
        
        # 原始频谱图
        self.original_spec_widget = SpectrogramWidget("原始梅尔频谱图")
        spectrograms_layout.addWidget(self.original_spec_widget)
        
        # 处理后频谱图
        self.processed_spec_widget = SpectrogramWidget("处理后梅尔频谱图")
        spectrograms_layout.addWidget(self.processed_spec_widget)
        
        main_layout.addLayout(spectrograms_layout)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # 音频播放器
        self.audio_player = AudioPlayerWidget()
        main_layout.addWidget(self.audio_player)
        
        # 打开输出文件夹按钮
        self.open_output_button = QPushButton("打开输出文件夹")
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.open_output_button.setEnabled(False)
        main_layout.addWidget(self.open_output_button)
        
        # 状态栏
        self.statusBar().showMessage('准备就绪，请拖放或选择MP3文件')
        
    def select_audio_file(self):
        """选择音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择MP3文件", "", "MP3文件 (*.mp3)"
        )
        if file_path:
            self.process_audio_file(file_path)
    
    def process_audio_file(self, file_path):
        """处理音频文件"""
        self.current_audio = file_path
        self.statusBar().showMessage(f'正在处理: {os.path.basename(file_path)}')
        
        # 显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # 禁用拖放区和选择按钮
        self.drop_area.setEnabled(False)
        self.select_file_button.setEnabled(False)
        
        # 创建并启动处理线程
        self.process_thread = AudioProcessingThread(file_path)
        self.process_thread.progress_signal.connect(self.update_progress)
        self.process_thread.result_signal.connect(self.update_audio_files)
        self.process_thread.finished_signal.connect(self.processing_finished)
        self.process_thread.error_signal.connect(self.show_error)
        self.process_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_audio_files(self, original_audio, processed_audio):
        """更新音频文件路径"""
        self.audio_player.set_audio_files(original_audio, processed_audio)
        self.processed_audio = processed_audio
    
    def processing_finished(self, output_dir, original_mel_spec, processed_mel_spec):
        """处理完成后的操作"""
        self.output_dir = output_dir
        
        # 显示梅尔频谱图
        self.original_spec_widget.set_image(original_mel_spec)
        self.processed_spec_widget.set_image(processed_mel_spec)
        
        # 启用拖放区和选择按钮
        self.drop_area.setEnabled(True)
        self.select_file_button.setEnabled(True)
        
        # 启用打开输出文件夹按钮
        self.open_output_button.setEnabled(True)
        
        # 更新状态栏
        self.statusBar().showMessage(f'处理完成: {os.path.basename(self.current_audio)}')
        
    def show_error(self, error_msg):
        """显示错误信息"""
        self.statusBar().showMessage(f'错误: {error_msg}')
        
        # 启用拖放区和选择按钮
        self.drop_area.setEnabled(True)
        self.select_file_button.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
    
    def open_output_folder(self):
        """打开输出文件夹"""
        if self.output_dir and os.path.exists(self.output_dir):
            # 根据操作系统打开文件夹
            if sys.platform == 'win32':
                os.startfile(self.output_dir)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{self.output_dir}"')
            else:  # linux
                os.system(f'xdg-open "{self.output_dir}"')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
