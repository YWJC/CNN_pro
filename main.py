import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MainApp:
    """主程序入口"""
    def __init__(self, root):
        self.root = root
        self.root.title("太赫兹光谱GADF图像分析系统")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="40")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        title_label = ttk.Label(self.main_frame, text="太赫兹光谱GADF图像分析系统", 
                               font=("SimHei", 18, "bold"))
        title_label.pack(pady=30)
        
        # 创建描述
        desc_label = ttk.Label(self.main_frame, text="该系统用于分析物质的太赫兹光谱GADF图像，支持多任务深度学习模型训练和性能分析。",
                             font=("SimHei", 12), justify=tk.CENTER)
        desc_label.pack(pady=20)
        
        # 创建按钮框架
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=40)
        
        # 模型训练按钮
        ttk.Button(button_frame, text="模型训练", command=self._open_train_gui, 
                  width=20).pack(side=tk.LEFT, padx=20)
        
        # 模型分析按钮
        ttk.Button(button_frame, text="模型分析", command=self._open_analysis_gui, 
                  width=20).pack(side=tk.RIGHT, padx=20)
    
    def _open_train_gui(self):
        """打开训练GUI"""
        try:
            from train_gui import main as train_main
            self.root.destroy()  # 关闭当前窗口
            train_main()
        except ImportError:
            messagebox.showerror("错误", "无法导入训练模块，请确保train_gui.py文件存在。")
    
    def _open_analysis_gui(self):
        """打开分析GUI"""
        try:
            from analysis_gui import main as analysis_main
            self.root.destroy()  # 关闭当前窗口
            analysis_main()
        except ImportError:
            messagebox.showerror("错误", "无法导入分析模块，请确保analysis_gui.py文件存在。")


def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()