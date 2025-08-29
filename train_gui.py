import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys

from train import train_model

class TrainGUI:
    """模型训练GUI界面"""
    def __init__(self, root):
        self.root = root
        self.root.title("深度学习模型训练")
        self.root.geometry("750x680")  # 稍微增加高度以容纳多选列表
        self.root.resizable(True, True)
        
        # 改进中文字体设置
        try:
            self.style = ttk.Style()
            # 确保中文能正常显示
            self.style.configure('.', font=('SimHei', 10))
        except:
            pass  # 如果字体不可用则跳过
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建参数配置区域
        self._create_param_frame()
        
        # 创建训练日志区域
        self._create_log_frame()
        
        # 创建按钮区域
        self._create_button_frame()
        
        # 训练线程
        self.train_thread = None
        self.is_training = False
        self.current_model_index = 0
        self.selected_models = []
    
    def _create_param_frame(self):
        """创建参数配置框架"""
        param_frame = ttk.LabelFrame(self.main_frame, text="训练参数配置", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        # 数据目录
        ttk.Label(param_frame, text="数据目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_dir_var = tk.StringVar(value="data")
        data_dir_entry = ttk.Entry(param_frame, textvariable=self.data_dir_var, width=50)
        data_dir_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Button(param_frame, text="浏览...", command=self._browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # 骨干网络选择 - 改为多选列表框
        ttk.Label(param_frame, text="骨干网络 (可多选):").grid(row=1, column=0, sticky=tk.NW, pady=5)
        
        # 可选择的骨干网络列表
        self.backbone_list = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2", "vgg16", "vgg19"]
        
        # 创建滚动框架
        list_frame = ttk.Frame(param_frame)
        list_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建多选列表框
        self.backbone_listbox = tk.Listbox(list_frame, selectmode='multiple', 
                                          exportselection=0, width=47, height=4)
        self.backbone_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 连接滚动条和列表框
        self.backbone_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.backbone_listbox.yview)
        
        # 添加选项到列表框
        for backbone in self.backbone_list:
            self.backbone_listbox.insert(tk.END, backbone)
        
        # 默认选中第一个选项
        self.backbone_listbox.select_set(0)
        
        # 训练轮次
        ttk.Label(param_frame, text="训练轮次:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=50).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 批次大小
        ttk.Label(param_frame, text="批次大小:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=50).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 学习率
        ttk.Label(param_frame, text="学习率:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(param_frame, textvariable=self.lr_var, width=50).grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # 权重衰减
        ttk.Label(param_frame, text="权重衰减:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.weight_decay_var = tk.StringVar(value="0.0001")
        ttk.Entry(param_frame, textvariable=self.weight_decay_var, width=50).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # 早停耐心值
        ttk.Label(param_frame, text="早停耐心值:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.patience_var = tk.StringVar(value="10")
        ttk.Entry(param_frame, textvariable=self.patience_var, width=50).grid(row=6, column=1, sticky=tk.W, pady=5)
    
    def _create_log_frame(self):
        """创建日志显示框架"""
        log_frame = ttk.LabelFrame(self.main_frame, text="训练日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建文本框用于显示日志
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 重定向标准输出到文本框
        sys.stdout = TextRedirector(self.log_text)
    
    def _create_button_frame(self):
        """创建按钮框架"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 确保开始训练按钮可见且可用
        self.start_button = ttk.Button(button_frame, text="开始训练", command=self._start_training, width=15)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # 停止训练按钮
        self.stop_button = ttk.Button(button_frame, text="停止训练", command=self._stop_training, state=tk.DISABLED, width=15)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # 退出按钮
        ttk.Button(button_frame, text="退出", command=self.root.destroy, width=10).pack(side=tk.RIGHT, padx=10)
    
    def _browse_data_dir(self):
        """浏览数据目录"""
        dir_path = filedialog.askdirectory(title="选择数据目录")
        if dir_path:
            self.data_dir_var.set(dir_path)
    
    def _start_training(self):
        """开始训练"""
        # 验证参数
        try:
            data_dir = self.data_dir_var.get()
            if not os.path.exists(data_dir):
                messagebox.showerror("错误", "数据目录不存在！")
                return
            
            # 获取选中的骨干网络
            selected_indices = self.backbone_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("错误", "请至少选择一个骨干网络！")
                return
            
            self.selected_models = [self.backbone_list[i] for i in selected_indices]
            num_epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            learning_rate = float(self.lr_var.get())
            weight_decay = float(self.weight_decay_var.get())
            patience = int(self.patience_var.get())
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数！")
            return
        
        # 禁用开始按钮，启用停止按钮
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        
        # 创建训练线程
        self.is_training = True
        self.current_model_index = 0
        
        self.train_thread = threading.Thread(target=self._run_multi_model_training, 
                                            args=(data_dir, num_epochs, batch_size, 
                                                  learning_rate, weight_decay, 
                                                  patience))
        self.train_thread.daemon = True
        self.train_thread.start()
    
    def _run_multi_model_training(self, data_dir, num_epochs, batch_size, learning_rate, 
                                 weight_decay, patience):
        """依次训练多个模型"""
        result_dirs = []
        
        try:
            total_models = len(self.selected_models)
            
            # 依次训练每个选中的模型
            for i, backbone_name in enumerate(self.selected_models):
                if not self.is_training:  # 检查是否已停止训练
                    self.log_text.insert(tk.END, f"\n训练已停止，未完成所有模型训练。\n")
                    break
                
                # 打印模型训练开始信息
                self.log_text.insert(tk.END, f"\n========= 开始训练模型 {i+1}/{total_models}: {backbone_name} =========\n")
                self.log_text.see(tk.END)
                
                # 训练单个模型
                result_dir = train_model(
                    data_dir=data_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    backbone_name=backbone_name,
                    patience=patience
                )
                
                if self.is_training:  # 再次检查是否已停止训练
                    result_dirs.append(result_dir)
                    self.log_text.insert(tk.END, f"\n========= 模型 {i+1}/{total_models}: {backbone_name} 训练完成 =========\n\n")
                    self.log_text.see(tk.END)
            
            # 如果所有模型都已训练完成
            if self.is_training and len(result_dirs) > 0:
                if len(result_dirs) == 1:
                    message = f"所有模型训练完成！结果保存至: {result_dirs[0]}"
                else:
                    message = "所有模型训练完成！结果分别保存至:\n"
                    for i, dir_path in enumerate(result_dirs):
                        message += f"{i+1}. {self.selected_models[i]}: {dir_path}\n"
                
                self.root.after(100, lambda: messagebox.showinfo("成功", message))
                
        except Exception as e:
            # 修复：使用默认参数方式传递异常变量e
            self.root.after(100, lambda e=e: messagebox.showerror("错误", f"训练过程中出错: {str(e)}"))
        finally:
            # 恢复按钮状态
            self.root.after(100, self._reset_ui)
    
    def _stop_training(self):
        """停止训练"""
        self.is_training = False
        self.log_text.insert(tk.END, "\n正在停止训练...\n")
        self.log_text.see(tk.END)
    
    def _reset_ui(self):
        """重置UI状态"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.is_training = False


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
    app = TrainGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()