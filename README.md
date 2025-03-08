# Colab-音頻分離

```python
import os
import sys
import subprocess
import time
import glob
import shutil
from pathlib import Path
from google.colab import files
from IPython.display import display, Audio, HTML, clear_output

class AudioSeparator:
    def __init__(self):
        self.root_dir = Path("/content")
        self.output_dir = self.root_dir / "separated_audio"
        self.temp_dir = self.root_dir / "temp_audio"
        
    def show_status(self, message, success=None):
        """顯示帶有圖標的狀態信息"""
        if success is None:
            icon = "ℹ️"
        elif success:
            icon = "✅"
        else:
            icon = "❌"
            
        print(f"{icon} {message}")
        
    def run_command(self, cmd, desc=None, check=True, show_output=False):
        """執行命令並返回結果"""
        if desc:
            self.show_status(desc)
        
        if show_output:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                      universal_newlines=True, bufsize=1)
            
            # 實時顯示輸出
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            return_code = process.wait()
            
            if check and return_code != 0:
                self.show_status(f"命令失敗，返回代碼: {return_code}", False)
                return False
                
            return True
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if check and result.returncode != 0:
                self.show_status(f"命令失敗: {cmd}\n{result.stderr}", False)
                return False
            
            return result
    
    def setup_environment(self):
        """設置執行環境"""
        self.show_status("🚀 配置處理環境...")
        
        # 創建目錄
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 安裝基本依賴
        self.show_status("安裝基本音頻處理工具...")
        self.run_command("apt-get update && apt-get install -y ffmpeg libsndfile1", check=False)
        
        # 安裝核心Python庫
        core_packages = [
            "torch torchaudio --no-deps",     # PyTorch 和 音頻處理
            "pydub",                          # 通用音頻處理
            "librosa soundfile",              # 高級音頻分析
        ]
        
        for package in core_packages:
            self.run_command(f"pip install -q {package}")
        
        # 嘗試安裝 Demucs（容錯）
        self.show_status("嘗試安裝 Demucs...")
        demucs_result = self.run_command("pip install -q demucs", check=False)
        
        if not demucs_result or isinstance(demucs_result, bool):
            self.show_status("標準安裝失敗，嘗試使用備用方式安裝 Demucs...")
            self.run_command("pip install -q -U demucs --no-deps", check=False)
            self.run_command("pip install -q diffq einops julius onnx", check=False)
            
        # 確認 ffmpeg 可用（這是最關鍵的依賴）
        ffmpeg_check = self.run_command("ffmpeg -version", check=False)
        if not ffmpeg_check:
            self.show_status("警告: ffmpeg 安裝可能有問題，嘗試修復...", False)
            self.run_command("apt-get install -y --reinstall ffmpeg", check=False)
        
        self.show_status("環境設置完成", True)
        return True
    
    def prepare_audio_file(self, input_path):
        """準備音頻文件（處理路徑和格式問題）"""
        original_path = Path(input_path)
        
        # 創建沒有空格的臨時文件名
        safe_filename = original_path.stem.replace(" ", "_") + original_path.suffix
        safe_path = self.temp_dir / safe_filename
        
        # 複製文件到臨時目錄
        self.show_status(f"準備音頻文件: {original_path.name}...")
        shutil.copy2(original_path, safe_path)
        
        # 檢查文件有效性
        check_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{str(safe_path)}"'
        probe_result = self.run_command(check_cmd, check=False)
        
        if not probe_result:
            self.show_status("檔案可能不是有效的音頻文件，嘗試修復...", False)
            fixed_path = self.temp_dir / f"fixed_{safe_filename}"
            fix_cmd = f'ffmpeg -y -i "{str(safe_path)}" -c:a libmp3lame -q:a 2 "{str(fixed_path)}"'
            fix_result = self.run_command(fix_cmd, check=False)
            
            if fix_result:
                safe_path = fixed_path
                self.show_status("音頻檔案已修復", True)
            else:
                self.show_status("無法修復音頻檔案", False)
        
        return safe_path
    
    def separate_with_demucs(self, input_file):
        """使用 Demucs 進行音頻分離"""
        # 使用各種模型嘗試
        models = [
            {"name": "基本分離", "cmd": f'demucs "{str(input_file)}" -o "{str(self.output_dir)}"'},
            {"name": "HTDemucs (兩聲部)", "cmd": f'demucs "{str(input_file)}" --two-stems=vocals -o "{str(self.output_dir)}"'},
            {"name": "MDX-Net", "cmd": f'demucs "{str(input_file)}" -n mdx_extra_q -o "{str(self.output_dir)}"'},
            {"name": "HTDemucs (四聲部)", "cmd": f'demucs "{str(input_file)}" -n htdemucs -o "{str(self.output_dir)}"'}
        ]
        
        for model in models:
            self.show_status(f"嘗試 {model['name']}...")
            start_time = time.time()
            
            initial_files = set(glob.glob(f"{self.output_dir}/**/*.*", recursive=True))
            result = self.run_command(model["cmd"], check=False, show_output=True)
            elapsed = time.time() - start_time
            
            # 檢查是否生成了新文件
            time.sleep(1)  # 等待文件系統更新
            current_files = set(glob.glob(f"{self.output_dir}/**/*.*", recursive=True))
            new_files = current_files - initial_files
            
            if new_files:
                self.show_status(f"{model['name']} 成功！處理時間: {elapsed:.1f}秒", True)
                return True
                
        return False
    
    def separate_with_ffmpeg(self, input_file):
        """使用 FFmpeg 進行基本音頻處理"""
        self.show_status("嘗試使用 FFmpeg 直接處理...")
        output_folder = self.output_dir / "ffmpeg_output"
        os.makedirs(output_folder, exist_ok=True)
        
        # 嘗試不同的 FFmpeg 操作
        operations = [
            {
                "name": "低通濾波器 (移除高頻)",
                "cmd": f'ffmpeg -i "{input_file}" -af "lowpass=f=4000" "{output_folder}/lowpass.mp3"'
            },
            {
                "name": "高通濾波器 (移除低頻)",
                "cmd": f'ffmpeg -i "{input_file}" -af "highpass=f=400" "{output_folder}/highpass.mp3"'
            },
            {
                "name": "中頻增強",
                "cmd": f'ffmpeg -i "{input_file}" -af "equalizer=f=1000:width_type=o:width=2:g=5" "{output_folder}/mid_boost.mp3"'
            },
            {
                "name": "立體聲分離",
                "cmd": f'ffmpeg -i "{input_file}" -map_channel 0.0.0 "{output_folder}/left.mp3" -map_channel 0.0.1 "{output_folder}/right.mp3"'
            }
        ]
        
        success_count = 0
        
        for op in operations:
            self.show_status(f"嘗試 {op['name']}...")
            result = self.run_command(op["cmd"], check=False)
            
            if result:
                success_count += 1
                
        if success_count > 0:
            self.show_status(f"FFmpeg 處理成功完成了 {success_count}/{len(operations)} 個操作", True)
            return True
        else:
            self.show_status("FFmpeg 處理失敗", False)
            return False
    
    def process_audio_file(self, input_file):
        """處理單個音頻文件"""
        # 先準備文件（處理路徑問題）
        safe_path = self.prepare_audio_file(input_file)
        if not safe_path:
            return False
            
        # 嘗試用 Demucs 處理
        self.show_status("嘗試使用 Demucs 分離...")
        if self.separate_with_demucs(safe_path):
            return True
            
        # 如果 Demucs 失敗，嘗試用 FFmpeg
        self.show_status("Demucs 處理失敗，嘗試 FFmpeg 基本處理...")
        if self.separate_with_ffmpeg(safe_path):
            return True
            
        # 都失敗了
        self.show_status(f"無法處理文件: {Path(input_file).name}", False)
        return False
    
    def preview_audio(self, file_path):
        """顯示音頻預覽"""
        try:
            display(Audio(str(file_path)))
            return True
        except Exception as e:
            self.show_status(f"無法預覽音頻: {str(e)}", False)
            return False
    
    def list_and_download_outputs(self, preview=True):
        """列出、預覽和下載結果"""
        self.show_status("檢查輸出文件...")
        
        # 尋找音頻文件
        extensions = ["wav", "mp3", "flac", "ogg"]
        output_files = []
        
        for ext in extensions:
            output_files.extend(glob.glob(f"{self.output_dir}/**/*.{ext}", recursive=True))
            
        if not output_files:
            self.show_status("未找到輸出文件", False)
            return False
        
        # 按目錄分組
        files_by_folder = {}
        for file_path in output_files:
            folder = os.path.dirname(file_path)
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            files_by_folder[folder].append(file_path)
        
        # 顯示和下載
        self.show_status(f"找到 {len(output_files)} 個輸出文件（{len(files_by_folder)} 個分組）", True)
        
        for folder, files_list in files_by_folder.items():
            folder_name = os.path.basename(folder)
            print(f"\n📁 {folder_name} ({len(files_list)} 個文件)")
            
            for file_path in sorted(files_list):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                
                print(f"  - 📄 {file_name} ({file_size:.2f} MB)")
                
                if preview:
                    self.preview_audio(file_path)
                
                # 下載文件
                files.download(file_path)
                
        return True
    
    def run_workflow(self):
        """執行完整工作流"""
        self.show_status("===== 音頻分離工具 =====")
        
        # 設置環境
        if not self.setup_environment():
            self.show_status("環境設置失敗，無法繼續", False)
            return False
            
        # 上傳文件
        self.show_status("請上傳音頻文件 (支持 .mp3, .wav, .flac 等格式)...")
        uploaded = files.upload()
        
        if not uploaded:
            self.show_status("未上傳文件", False)
            return False
        
        # 處理文件
        success_count = 0
        for filename, content in uploaded.items():
            input_path = self.root_dir / filename
            
            # 確保獲得文件內容
            if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                with open(input_path, 'wb') as f:
                    f.write(content)
                    
            # 處理文件
            if self.process_audio_file(input_path):
                success_count += 1
                
        # 列出和下載結果
        if success_count > 0:
            self.list_and_download_outputs()
            self.show_status(f"成功處理 {success_count}/{len(uploaded)} 個文件", True)
        else:
            self.show_status("所有文件處理失敗", False)
            
        self.show_status("===== 處理完成 =====")
        return success_count > 0

# 執行工作流程
processor = AudioSeparator()
processor.run_workflow()
```
