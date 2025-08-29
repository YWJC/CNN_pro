import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analysis import analyze_model, compare_models

# 设置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "NSimSun", "KaiTi", "FangSong"]
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "NSimSun", "KaiTi", "FangSong"]
plt.rcParams["axes.unicode_minus"] = False
class AnalysisGUI:
    """模型分析GUI界面"""
    def __init__(self, root):
        self.root = root
        self.root.title("模型性能分析")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 存储选择的模型路径
        self.model_paths = []
        self.backbone_names = []
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建配置区域
        self._create_config_frame()
        
        # 创建结果显示区域
        self._create_result_frame()
        
        # 创建按钮区域
        self._create_button_frame()
        
        # 分析线程
        self.analysis_thread = None
    
    def _create_config_frame(self):
        """创建配置框架"""
        config_frame = ttk.LabelFrame(self.main_frame, text="分析配置", padding="10")
        config_frame.pack(fill=tk.X, pady=10)
        
        # 数据目录
        ttk.Label(config_frame, text="数据目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_dir_var = tk.StringVar(value="data")
        data_dir_entry = ttk.Entry(config_frame, textvariable=self.data_dir_var, width=60)
        data_dir_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Button(config_frame, text="浏览...", command=self._browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # 模型文件列表
        ttk.Label(config_frame, text="模型文件:").grid(row=1, column=0, sticky=tk.NW, pady=5)
        
        # 创建模型列表框架
        model_list_frame = ttk.Frame(config_frame)
        model_list_frame.grid(row=1, column=1, sticky=tk.NSEW, pady=5)
        
        # 模型列表
        self.model_listbox = tk.Listbox(model_list_frame, width=60, height=5)
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(model_list_frame, command=self.model_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.config(yscrollcommand=scrollbar.set)
        
        # 按钮
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Button(button_frame, text="添加模型", command=self._add_model).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="移除选中", command=self._remove_model).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="清空列表", command=self._clear_models).pack(fill=tk.X, pady=2)
    
    def _create_result_frame(self):
        """创建结果显示框架"""
        result_frame = ttk.LabelFrame(self.main_frame, text="分析结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 日志标签页
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="分析日志")
        
        # 创建文本框用于显示日志
        self.log_text = tk.Text(self.log_tab, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.log_tab, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 图表标签页
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="性能图表")
        
        # 重定向标准输出到文本框
        sys.stdout = TextRedirector(self.log_text)
    
    def _create_button_frame(self):
        """创建按钮框架"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 分析单个模型按钮
        ttk.Button(button_frame, text="分析单个模型", command=self._analyze_single_model).pack(side=tk.LEFT, padx=5)
        
        # 比较多个模型按钮
        ttk.Button(button_frame, text="比较多个模型", command=self._compare_models).pack(side=tk.LEFT, padx=5)
        
        # 打开结果目录按钮
        ttk.Button(button_frame, text="打开结果目录", command=self._open_result_dir).pack(side=tk.LEFT, padx=5)
        
        # 退出按钮
        ttk.Button(button_frame, text="退出", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 存储最后分析的结果目录
        self.last_result_dir = None
    
    def _browse_data_dir(self):
        """浏览数据目录"""
        dir_path = filedialog.askdirectory(title="选择数据目录")
        if dir_path:
            self.data_dir_var.set(dir_path)
    
    def _add_model(self):
        """添加模型文件"""
        file_paths = filedialog.askopenfilenames(title="选择模型文件", filetypes=[("PyTorch模型", "*.pth")])
        if file_paths:
            for file_path in file_paths:
                if file_path not in self.model_paths:
                    self.model_paths.append(file_path)
                    
                    # 获取骨干网络名称（从文件名推断）
                    file_name = os.path.basename(os.path.dirname(file_path))
                    
                    # 改进的骨干网络名称提取逻辑
                    if 'efficientnet' in file_name:
                        # 提取完整的efficientnet型号，如efficientnet_b0, efficientnet_b1等
                        backbone_name = 'efficientnet_b' + ''.join([c for c in file_name.split('efficientnet_b')[1].split('_')[0] if c.isdigit()])
                    elif 'mobilenet' in file_name:
                        # 处理mobilenet型号
                        backbone_name = 'mobilenet_v2'
                    elif 'vgg' in file_name:
                        # 处理vgg型号
                        vgg_num = ''.join([c for c in file_name.split('vgg')[1].split('_')[0] if c.isdigit()])
                        backbone_name = f'vgg{vgg_num}' if vgg_num else 'vgg16'
                    elif 'resnet' in file_name:
                        # 处理resnet型号
                        resnet_num = ''.join([c for c in file_name.split('resnet')[1].split('_')[0] if c.isdigit()])
                        backbone_name = f'resnet{resnet_num}' if resnet_num else 'resnet18'
                    else:
                        backbone_name = 'resnet18'  # 默认值
                    
                    self.backbone_names.append(backbone_name)
                    
                    # 在列表中显示文件名
                    self.model_listbox.insert(tk.END, os.path.basename(file_path))
    
    def _remove_model(self):
        """移除选中的模型"""
        selected_indices = self.model_listbox.curselection()
        if selected_indices:
            # 从后往前删除，避免索引变化
            for i in sorted(selected_indices, reverse=True):
                self.model_listbox.delete(i)
                del self.model_paths[i]
                del self.backbone_names[i]
    
    def _clear_models(self):
        """清空模型列表"""
        self.model_listbox.delete(0, tk.END)
        self.model_paths = []
        self.backbone_names = []
    
    def _analyze_single_model(self):
        """分析单个模型"""
        if len(self.model_paths) != 1:
            messagebox.showinfo("提示", "请选择一个模型进行分析")
            return
        
        # 验证数据目录
        data_dir = self.data_dir_var.get()
        if not os.path.exists(data_dir):
            messagebox.showerror("错误", "数据目录不存在！")
            return
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        
        # 创建分析线程
        self.analysis_thread = threading.Thread(target=self._run_analyze, 
                                               args=(self.model_paths[0], data_dir, self.backbone_names[0]))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def _compare_models(self):
        """比较多个模型"""
        if len(self.model_paths) < 2:
            messagebox.showinfo("提示", "请至少选择两个模型进行比较")
            return
        
        # 验证数据目录
        data_dir = self.data_dir_var.get()
        if not os.path.exists(data_dir):
            messagebox.showerror("错误", "数据目录不存在！")
            return
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        
        # 创建分析线程
        self.analysis_thread = threading.Thread(target=compare_models, 
                                               args=(self.model_paths, data_dir, self.backbone_names))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def _run_analyze(self, model_path, data_dir, backbone_name):
        """运行模型分析"""
        try:
            result_dir = analyze_model(model_path, data_dir, backbone_name)
            self.last_result_dir = result_dir
            self.root.after(100, lambda: messagebox.showinfo("成功", f"模型分析完成！结果保存在: {result_dir}"))
        except Exception as e:
            # 使用默认参数方式传递e变量
            self.root.after(100, lambda e=e: messagebox.showerror("错误", f"分析过程中出错: {str(e)}"))
    
    def _open_result_dir(self):
        """打开结果目录"""
        if self.last_result_dir and os.path.exists(self.last_result_dir):
            # 根据操作系统打开文件夹
            import subprocess
            if sys.platform.startswith('win'):
                os.startfile(self.last_result_dir)
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', self.last_result_dir])
            else:
                subprocess.Popen(['xdg-open', self.last_result_dir])
        else:
            messagebox.showinfo("提示", "没有找到可打开的结果目录")


class TextRedirector:
    """用于将标准输出重定向到Tkinter文本框"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
    
    def write(self, string):
        # 在主线程中更新UI
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # 滚动到最后
    
    def flush(self):
        pass  # 必须实现flush方法


def main():
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()