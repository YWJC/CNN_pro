import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from utils.data_loader import get_data_loaders
from utils.early_stopping import EarlyStopping
from models.base_model import MultiTaskModel

# 设置中文字体
# 替换当前的字体配置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "NSimSun", "KaiTi", "FangSong"]
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "NSimSun", "KaiTi", "FangSong"]
plt.rcParams["axes.unicode_minus"] = False

# 添加R方计算函数
def compute_r2(y_true, y_pred):
    """计算R方值
    R² = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    """
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    
    # 避免除零错误
    if ss_total == 0:
        return torch.tensor(1.0, device=y_true.device)
        
    r2 = 1 - (ss_residual / ss_total)
    return r2


def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001,
                weight_decay=1e-4, backbone_name='resnet18', patience=10):
    """训练多任务模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    model = MultiTaskModel(backbone_name=backbone_name).to(device)
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失
    criterion_reg = nn.MSELoss()  # 回归损失
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器 - 移除verbose参数
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=patience)
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join('results', f'{backbone_name}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    # 训练历史 - 添加R方指标
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cls_acc': [],
        'val_cls_acc': [],
        'train_reg_mae': [],
        'val_reg_mae': [],
        'train_reg_r2': [],  # 添加训练R方
        'val_reg_r2': []     # 添加验证R方
    }
    
    # 开始训练
    print(f"开始训练 {backbone_name} 模型...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_correct = 0
        train_reg_errors = 0
        
        # 用于计算R方的累积变量
        all_train_preds = []
        all_train_labels = []
        
        # 批次进度跟踪
        batch_count = len(train_loader)
        
        # 训练循环
        for batch_idx, (images, cls_labels, reg_labels) in enumerate(train_loader):
            images, cls_labels, reg_labels = images.to(device), cls_labels.to(device), reg_labels.to(device)
            
            # 前向传播
            cls_output, reg_output = model(images)
            
            # 计算损失
            loss_cls = criterion_cls(cls_output, cls_labels)
            loss_reg = criterion_reg(reg_output.squeeze(), reg_labels)
            # 总损失（分类和回归的权重可以调整）
            loss = loss_cls + loss_reg
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            
            # 分类准确率
            _, preds = torch.max(cls_output, 1)
            train_cls_correct += torch.sum(preds == cls_labels.data)
            
            # 回归误差 (MAE)
            train_reg_errors += torch.sum(torch.abs(reg_output.squeeze() * 1000 - reg_labels * 1000))
            
            # 收集所有预测和真实值用于计算R方
            all_train_preds.append(reg_output.squeeze().cpu().detach())
            all_train_labels.append(reg_labels.cpu().detach())
            
            # 显示详细进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
                progress = (batch_idx + 1) / batch_count * 100
                current_batch_loss = loss.item()
                current_batch_acc = (torch.sum(preds == cls_labels.data) / len(cls_labels)).item()
                current_batch_mae = (torch.sum(torch.abs(reg_output.squeeze() * 1000 - reg_labels * 1000)) / len(reg_labels)).item()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{batch_count} ({progress:.1f}%): Loss={current_batch_loss:.4f}, Acc={current_batch_acc:.4f}, MAE={current_batch_mae:.2f}nM")
        
        # 计算训练R方
        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)
        train_r2 = compute_r2(all_train_labels, all_train_preds)
        
        # 验证循环
        model.eval()
        val_loss = 0.0
        val_cls_correct = 0
        val_reg_errors = 0
        
        # 用于计算R方的累积变量
        all_val_preds = []
        all_val_labels = []
        
        # 验证进度跟踪
        val_batch_count = len(val_loader)
        
        with torch.no_grad():
            for val_batch_idx, (images, cls_labels, reg_labels) in enumerate(val_loader):
                images, cls_labels, reg_labels = images.to(device), cls_labels.to(device), reg_labels.to(device)
                
                cls_output, reg_output = model(images)
                
                loss_cls = criterion_cls(cls_output, cls_labels)
                loss_reg = criterion_reg(reg_output.squeeze(), reg_labels)
                loss = loss_cls + loss_reg
                
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(cls_output, 1)
                val_cls_correct += torch.sum(preds == cls_labels.data)
                
                val_reg_errors += torch.sum(torch.abs(reg_output.squeeze() * 1000 - reg_labels * 1000))
                
                # 收集所有预测和真实值用于计算R方
                all_val_preds.append(reg_output.squeeze().cpu())
                all_val_labels.append(reg_labels.cpu())
                
                # 显示验证进度
                if (val_batch_idx + 1) % 10 == 0 or (val_batch_idx + 1) == val_batch_count:
                    val_progress = (val_batch_idx + 1) / val_batch_count * 100
                    print(f"验证中... {val_progress:.1f}%")
        
        # 计算验证R方
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)
        val_r2 = compute_r2(all_val_labels, all_val_preds)
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_cls_acc = train_cls_correct.double() / len(train_loader.dataset)
        val_cls_acc = val_cls_correct.double() / len(val_loader.dataset)
        
        train_reg_mae = train_reg_errors.double() / len(train_loader.dataset)
        val_reg_mae = val_reg_errors.double() / len(val_loader.dataset)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_cls_acc'].append(train_cls_acc.item())
        history['val_cls_acc'].append(val_cls_acc.item())
        history['train_reg_mae'].append(train_reg_mae.item())
        history['val_reg_mae'].append(val_reg_mae.item())
        history['train_reg_r2'].append(train_r2.item())  # 记录训练R方
        history['val_reg_r2'].append(val_r2.item())      # 记录验证R方
        
        # 打印 epoch 结果 - 添加R方指标
        print(f"Epoch {epoch+1}/{num_epochs}: \n" \
              f"  训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f} \n" \
              f"  训练分类准确率: {train_cls_acc:.4f}, 验证分类准确率: {val_cls_acc:.4f} \n" \
              f"  训练回归MAE: {train_reg_mae:.2f}nM, 验证回归MAE: {val_reg_mae:.2f}nM \n" \
              f"  训练回归R方: {train_r2:.4f}, 验证回归R方: {val_r2:.4f}")
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"早停机制触发，在第 {epoch+1} 轮停止训练")
            break
        
        # 保存最佳模型
        if avg_val_loss <= early_stopping.val_loss_min:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(result_dir, 'best_model.pth'))
    
    # 训练结束
    end_time = time.time()
    print(f"训练完成！耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 保存训练历史
    torch.save(history, os.path.join(result_dir, 'training_history.pth'))
    
    # 绘制训练历史
    plot_training_history(history, result_dir)
    
    return result_dir

# 修改绘图函数以包含R方曲线
def plot_training_history(history, save_dir):
    """绘制训练历史图表"""
    plt.figure(figsize=(15, 15))
    
    # 绘制损失曲线
    plt.subplot(4, 1, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制分类准确率
    plt.subplot(4, 1, 2)
    plt.plot(history['train_cls_acc'], label='训练准确率')
    plt.plot(history['val_cls_acc'], label='验证准确率')
    plt.title('分类准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    # 绘制回归MAE
    plt.subplot(4, 1, 3)
    plt.plot(history['train_reg_mae'], label='训练MAE')
    plt.plot(history['val_reg_mae'], label='验证MAE')
    plt.title('回归MAE (nM)')
    plt.xlabel('轮次')
    plt.ylabel('MAE (nM)')
    plt.legend()
    
    # 添加R方曲线
    plt.subplot(4, 1, 4)
    plt.plot(history['train_reg_r2'], label='训练R方')
    plt.plot(history['val_reg_r2'], label='验证R方')
    plt.title('回归R方')
    plt.xlabel('轮次')
    plt.ylabel('R方值')
    plt.ylim(-0.5, 1.0)  # 设置合理的Y轴范围
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

if __name__ == '__main__':
    # 默认参数
    data_dir = 'data'
    
    # 训练模型
    result_dir = train_model(
        data_dir=data_dir,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        backbone_name='resnet18'
    )
    
    print(f"结果保存至: {result_dir}")