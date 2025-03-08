未修正版本

```python
import os
import sys
import subprocess
import time
import glob
import shutil
import random
import threading
from pathlib import Path
from google.colab import files
from IPython.display import display, Audio, HTML, clear_output
from tqdm.notebook import tqdm  # 引入tqdm進行進度顯示

class AudioSeparator:
    def __init__(self):
        self.root_dir = Path("/content")
        self.output_dir = self.root_dir / "separated_audio"
        self.temp_dir = self.root_dir / "temp_audio"
        self.mp3_dir = self.output_dir / "mp3_versions"
        self.convert_to_mp3 = True  # 預設啟用MP3轉換
        self.mp3_bitrate = "192k"   # 預設MP3比特率
        self.setup_keep_alive()     # 初始化時啟動keep_alive線程

    def setup_keep_alive(self):
        """設置保持控制台活躍的線程"""
        def keep_alive():
            while True:
                braille_char = chr(random.randint(0x2800, 0x28FF))
                print(braille_char, end=" ", flush=True)
                time.sleep(60)  # 每分鐘輸出一次

        keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
        keep_alive_thread.start()

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

            # 將輸出重定向到tqdm進行顯示，避免直接print造成進度條問題
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line.strip())
                # 檢查此行是否為進度線
                if "%" in line and "[" in line and "]" in line:
                    # 提取百分比
                    try:
                        percent = int(float(line.split("%")[0].strip()))
                        # 使用\r覆蓋當前行來顯示乾淨的進度
                        tqdm.write(f"\r進度: {percent}% 完成", end="")
                    except:
                        tqdm.write(line, end="")
                else:
                    tqdm.write(line, end="")

            process.stdout.close()
            return_code = process.wait()
            tqdm.write("")  # 添加換行

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
        os.makedirs(self.mp3_dir, exist_ok=True)

        # 安裝基本依賴 (添加tqdm)
        self.show_status("安裝基本音頻處理工具...")
        self.run_command("apt-get update && apt-get install -y ffmpeg libsndfile1", check=False)

        # 安裝核心Python庫
        core_packages = [
            "torch torchaudio --no-deps",     # PyTorch 和 音頻處理
            "pydub",                          # 通用音頻處理
            "librosa soundfile",              # 高級音頻分析
            "tqdm",                           # 進度條顯示
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

            # 使用進度條顯示
            with tqdm(total=100, desc=f"{model['name']} 處理進度", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                # 在新線程中更新進度條
                def update_progress():
                    last_progress = 0
                    while pbar.n < 100:
                        # 根據經過的時間推算進度
                        elapsed = time.time() - start_time
                        # 假設處理時間大約需要3分鐘完成
                        estimated_total_time = 180
                        progress = min(99, int((elapsed / estimated_total_time) * 100))
                        if progress > last_progress:
                            pbar.update(progress - last_progress)
                            last_progress = progress
                        time.sleep(1)

                progress_thread = threading.Thread(target=update_progress)
                progress_thread.daemon = True
                progress_thread.start()

                # 執行命令
                result = self.run_command(model["cmd"], check=False, show_output=True)

                # 完成進度條
                pbar.n = 100
                pbar.refresh()

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

        # 使用tqdm顯示總體進度
        with tqdm(total=len(operations), desc="FFmpeg處理進度") as pbar:
            for op in operations:
                self.show_status(f"嘗試 {op['name']}...")
                result = self.run_command(op["cmd"], check=False)

                if result:
                    success_count += 1

                pbar.update(1)

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

    def convert_files_to_mp3(self, file_list):
        """將列表中的文件轉換為MP3格式"""
        if not self.convert_to_mp3:
            return [], {}

        self.show_status(f"開始轉換文件為MP3格式 (比特率: {self.mp3_bitrate})...")
        converted_files = []
        file_size_comparison = {}

        # 使用tqdm進度條
        with tqdm(total=len(file_list), desc="MP3轉換進度") as pbar:
            for source_file in file_list:
                source_path = Path(source_file)
                if source_path.suffix.lower() == '.mp3':
                    # 如果已經是MP3，直接複製
                    target_path = self.mp3_dir / source_path.name
                    shutil.copy2(source_file, target_path)
                    converted_files.append(str(target_path))
                    pbar.update(1)
                    continue

                # 創建與原目錄結構相似的子目錄
                rel_folder = os.path.relpath(os.path.dirname(source_file), str(self.output_dir))
                if rel_folder != '.':
                    target_folder = self.mp3_dir / rel_folder
                    os.makedirs(target_folder, exist_ok=True)
                else:
                    target_folder = self.mp3_dir

                # 生成MP3文件名
                target_name = source_path.stem + ".mp3"
                target_path = target_folder / target_name

                # 轉換為MP3
                cmd = f'ffmpeg -y -i "{source_file}" -codec:a libmp3lame -b:a {self.mp3_bitrate} "{target_path}"'
                result = self.run_command(cmd, check=False)

                if result and os.path.exists(target_path):
                    converted_files.append(str(target_path))

                    # 計算原文件和MP3的大小差異
                    orig_size = os.path.getsize(source_file) / (1024 * 1024)  # MB
                    mp3_size = os.path.getsize(target_path) / (1024 * 1024)   # MB
                    savings = orig_size - mp3_size
                    savings_percent = (savings / orig_size) * 100 if orig_size > 0 else 0

                    file_size_comparison[target_name] = {
                        'original': f"{orig_size:.2f} MB",
                        'mp3': f"{mp3_size:.2f} MB",
                        'savings': f"{savings:.2f} MB ({savings_percent:.1f}%)"
                    }

                pbar.update(1)

        # 顯示總體節省情況
        if converted_files:
            total_orig = sum(os.path.getsize(f) for f in file_list) / (1024 * 1024)
            total_mp3 = sum(os.path.getsize(f) for f in converted_files) / (1024 * 1024)
            total_savings = total_orig - total_mp3
            savings_percent = (total_savings / total_orig) * 100 if total_orig > 0 else 0

            self.show_status(f"MP3轉換完成! 從 {total_orig:.2f} MB 減小到 {total_mp3:.2f} MB，節省 {total_savings:.2f} MB ({savings_percent:.1f}%)", True)
        else:
            self.show_status("沒有文件可以轉換為MP3", False)

        return converted_files, file_size_comparison

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

        # 是否轉換為MP3
        if self.convert_to_mp3:
            mp3_files, size_comparison = self.convert_files_to_mp3(output_files)

            # 如果MP3轉換成功，使用MP3文件
            if mp3_files:
                self.show_status("將使用MP3版本進行下載 (較小的檔案大小)", True)
                download_files = mp3_files
            else:
                self.show_status("MP3轉換失敗，使用原始文件", False)
                download_files = output_files
        else:
            self.show_status("使用原始格式文件", True)
            download_files = output_files

        # 按目錄分組
        files_by_folder = {}
        for file_path in download_files:
            folder = os.path.dirname(file_path)
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            files_by_folder[folder].append(file_path)

        # 顯示和下載
        self.show_status(f"找到 {len(download_files)} 個輸出文件（{len(files_by_folder)} 個分組）", True)

        for folder, files_list in files_by_folder.items():
            folder_name = os.path.basename(folder)
            print(f"\n📁 {folder_name} ({len(files_list)} 個文件)")

            for file_path in sorted(files_list):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)

                size_info = f"({file_size:.2f} MB)"
                if self.convert_to_mp3 and file_name in size_comparison:
                    size_info += f" - 節省: {size_comparison[file_name]['savings']}"

                print(f"  - 📄 {file_name} {size_info}")

                if preview:
                    self.preview_audio(file_path)

                # 下載文件
                files.download(file_path)

        return True

    def configure_mp3_settings(self):
        """配置MP3轉換設置"""
        print("\n===== MP3 轉換設置 =====")
        print("將輸出文件轉換為MP3可顯著減少文件大小，加快下載速度。")

        convert_input = input("是否要轉換為MP3格式? (y/n，默認: y): ").strip().lower()
        self.convert_to_mp3 = convert_input != "n"  # 預設啟用

        if self.convert_to_mp3:
            print("\n可用的MP3比特率:")
            print("  - 128k (較小文件，較低音質)")
            print("  - 192k (平衡大小與音質)")
            print("  - 256k (較好音質，較大文件)")
            print("  - 320k (最高音質，最大文件)")

            bitrate_input = input("選擇MP3比特率 (默認: 192k): ").strip().lower()
            valid_bitrates = ["128k", "192k", "256k", "320k"]
            self.mp3_bitrate = bitrate_input if bitrate_input in valid_bitrates else "192k"

            print(f"已設置轉換為MP3，比特率: {self.mp3_bitrate}")
        else:
            print("將使用原始音頻格式（可能較大）")

    def run_workflow(self):
        """執行完整工作流"""
        self.show_status("===== 音頻分離工具 =====")

        # 設置環境
        if not self.setup_environment():
            self.show_status("環境設置失敗，無法繼續", False)
            return False

        # 配置MP3轉換設置
        self.configure_mp3_settings()

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

```
> 用戶分享了一個非常長的 Python 代碼文件，這似乎是一個複雜的音頻處理器，適用於 Google Colab。這段代碼是一個名為 `AudioSeparator` 的類，使用 Demucs 和 FFmpeg 等工具執行音頻分離任務。它有許多方法用於音頻處理工作流程的不同階段。
> 
> 根據之前的對話，用戶在詢問如何在 Google Colab 筆記本中管理長代碼，當代碼變得難以維護時。他們分享這段代碼作為他們所面對的長而複雜的代碼的例子。
> 
> 用戶似乎希望獲得有關如何更好地結構化這段代碼以提高可維護性的建議。由於這段代碼相當長且複雜，我將分析它並提供具體的重構和改善可維護性的建議。
> 
> 讓我分析這段代碼的主要問題並提供具體建議：
> 
> 1. **模組化**：
>    - 這個類非常大，有很多不同的責任
>    - 它處理 UI、環境設置、文件操作、音頻處理等
>    - 需要將其拆分為更小、更專注的模組
> 
> 2. **關注點分離**：
>    - UI/顯示邏輯與處理邏輯混合在一起
>    - 文件操作分散在不同的方法中
>    - 配置、處理和輸出處理應該分開
> 
> 3. **代碼組織**：
>    - 所有內容都在一個龐大的類中
>    - 一些方法相當長（例如 setup_environment, convert_files_to_mp3）
>    - 不同功能區域之間沒有明確的區分
> 
> 我將提供如何重組這段代碼以提高可維護性的建議，並給出具體的重構示例。
```

# 如何重構長程式碼：以AudioSeparator為例

針對您分享的這段長程式碼，以下是具體的重構建議，可以大幅提高其可維護性：

## 1. 按功能拆分為多個模組文件

將巨大的`AudioSeparator`類拆分為多個專注於單一責任的模組：

```
audio_separator/
│
├── __init__.py                # 導出主要類和函數
├── core.py                    # 核心協調類
├── environment.py             # 環境設置相關
├── processors/
│   ├── __init__.py
│   ├── demucs_processor.py    # Demucs處理邏輯
│   ├── ffmpeg_processor.py    # FFmpeg處理邏輯
│
├── utils/
│   ├── __init__.py
│   ├── file_operations.py     # 文件操作相關功能  
│   ├── command_runner.py      # 命令執行功能
│   ├── ui.py                  # 進度顯示和UI相關功能
│
└── converters/
    ├── __init__.py
    └── mp3_converter.py       # MP3轉換邏輯
```

## 2. 重構核心類結構

例如，將環境設置邏輯抽離為獨立模組：

```python
# environment.py
import os
import subprocess
from pathlib import Path

class EnvironmentSetup:
    def __init__(self, root_dir=Path("/content")):
        self.root_dir = root_dir
        self.output_dir = self.root_dir / "separated_audio"
        self.temp_dir = self.root_dir / "temp_audio"
        self.mp3_dir = self.output_dir / "mp3_versions"
        
    def create_directories(self):
        """創建必要的目錄"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.mp3_dir, exist_ok=True)
        
    def install_dependencies(self, show_status_callback=print):
        """安裝所需的依賴"""
        show_status_callback("安裝基本音頻處理工具...")
        # 安裝依賴的代碼...
        
    def setup(self, show_status_callback=print):
        """完整設置環境"""
        show_status_callback("🚀 配置處理環境...")
        self.create_directories()
        return self.install_dependencies(show_status_callback)
```

## 3. 實用的命令執行器

將命令執行邏輯抽離出來：

```python
# utils/command_runner.py
import subprocess
from tqdm.notebook import tqdm

class CommandRunner:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback or (lambda msg, success=None: None)
        
    def run(self, cmd, desc=None, check=True, show_output=False):
        """執行命令並返回結果"""
        if desc:
            self.status_callback(desc)
            
        # 命令執行邏輯...
        
        return result
```

## 4. 使用介面抽象化處理器

創建處理器的抽象介面，使添加新的處理方法更容易：

```python
# processors/base_processor.py
from abc import ABC, abstractmethod

class AudioProcessor(ABC):
    @abstractmethod
    def process(self, input_file, output_dir):
        """處理音頻文件"""
        pass

# processors/demucs_processor.py
from .base_processor import AudioProcessor

class DemucsProcessor(AudioProcessor):
    def __init__(self, command_runner, status_callback=None):
        self.command_runner = command_runner
        self.status_callback = status_callback
        
    def process(self, input_file, output_dir):
        """使用Demucs處理音頻文件"""
        # Demucs處理邏輯...
```

## 5. 使用配置類簡化參數管理

```python
# config.py
class AudioSeparatorConfig:
    def __init__(self):
        self.convert_to_mp3 = True
        self.mp3_bitrate = "192k"
        self.preview_audio = True
        
    def configure_from_user_input(self):
        """從用戶輸入獲取配置"""
        print("\n===== MP3 轉換設置 =====")
        # 用戶配置邏輯...
```

## 6. 在Colab中使用的簡化導入方式

在Colab中可以使用以下方式來更好地組織和導入這些模組：

```python
# 在Colab中創建模組結構
%%writefile audio_separator/core.py
# 這裡放入核心類代碼...

%%writefile audio_separator/environment.py
# 這裡放入環境設置代碼...

# 其他模組...

# 然後在主筆記本中導入
from audio_separator.core import AudioSeparator
processor = AudioSeparator()
processor.run_workflow()
```

## 7. 使用工廠模式選擇處理器

```python
# processors/factory.py
from .demucs_processor import DemucsProcessor
from .ffmpeg_processor import FFmpegProcessor

class ProcessorFactory:
    @staticmethod
    def create_processor(processor_type, command_runner, status_callback=None):
        if processor_type == "demucs":
            return DemucsProcessor(command_runner, status_callback)
        elif processor_type == "ffmpeg":
            return FFmpegProcessor(command_runner, status_callback)
        else:
            raise ValueError(f"未知的處理器類型: {processor_type}")
```

## 實施步驟

1. **首先拆分最基本的工具類**，如CommandRunner、UI類等
2. **創建基本的目錄結構**，將代碼分散到不同的文件中
3. **逐步重構**每個主要功能區域，一次專注於一個部分
4. **添加適當的單元測試**，確保重構過程中不會引入錯誤
5. **更新主工作流程**，使用新的模組化結構

這種重構不僅可以讓代碼更容易維護，還可以使功能擴展變得更簡單，例如添加新的音頻處理器或支持新的格式。