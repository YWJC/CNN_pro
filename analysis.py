import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, r2_score
import seaborn as sns

from utils.data_loader import get_data_loaders
from models.base_model import MultiTaskModel

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def analyze_model(model_path, data_dir, backbone_name='resnet18'):
    """分析模型性能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    _, _, test_loader, class_names = get_data_loaders(data_dir)
    
    # 创建模型并加载权重
    model = MultiTaskModel(backbone_name=backbone_name).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 存储预测结果
    all_cls_labels = []
    all_cls_preds = []
    all_reg_labels = []
    all_reg_preds = []
    all_substances = []  # 存储每个样本的物质类别
    
    with torch.no_grad():
        for images, cls_labels, reg_labels in test_loader:
            images = images.to(device)
            
            # 获取预测结果
            cls_preds, reg_preds = model.predict(images)
            
            # 保存结果
            all_cls_labels.extend(cls_labels.numpy())
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_reg_labels.extend((reg_labels * 1000).numpy())  # 还原为原始浓度
            all_reg_preds.extend(reg_preds.cpu().numpy())
            
            # 记录每个样本的物质类别
            for label in cls_labels.numpy():
                all_substances.append(class_names[label])
    
    # 创建结果保存目录
    result_dir = os.path.dirname(model_path)
    
    # 1. 分类性能分析
    print("\n分类性能分析:")
    cm = confusion_matrix(all_cls_labels, all_cls_preds)
    class_names_list = [class_names[i] for i in range(len(class_names))]
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 打印分类报告
    print("分类报告:")
    print(classification_report(all_cls_labels, all_cls_preds, target_names=class_names_list))
    
    # 2. 回归性能分析
    print("\n回归性能分析:")
    # 总体回归性能
    mae = np.mean(np.abs(np.array(all_reg_preds) - np.array(all_reg_labels)))
    mse = np.mean((np.array(all_reg_preds) - np.array(all_reg_labels)) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_reg_labels, all_reg_preds)
    
    print(f"总体回归性能:\n"
          f"  MAE: {mae:.2f}nM\n"
          f"  MSE: {mse:.2f}\n"
          f"  RMSE: {rmse:.2f}nM\n"
          f"  R²: {r2:.4f}")
    
    # 绘制总体回归散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_reg_labels, all_reg_preds, alpha=0.5)
    plt.plot([min(all_reg_labels), max(all_reg_labels)], 
             [min(all_reg_labels), max(all_reg_labels)], 'r--', lw=2)
    plt.title(f'总体回归性能 (MAE: {mae:.2f}nM, R²: {r2:.4f})')
    plt.xlabel('真实浓度 (nM)')
    plt.ylabel('预测浓度 (nM)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'overall_regression.png'))
    plt.close()
    
    # 3. 按物质类别分析回归性能
    print("\n按物质类别回归性能分析:")
    
    # 按物质分类结果
    substance_results = {}
    for sub, true, pred in zip(all_substances, all_reg_labels, all_reg_preds):
        if sub not in substance_results:
            substance_results[sub] = {'true': [], 'pred': []}
        substance_results[sub]['true'].append(true)
        substance_results[sub]['pred'].append(pred)
    
    # 为每个物质绘制回归散点图
    plt.figure(figsize=(15, 5))
    
    for i, (sub, results) in enumerate(substance_results.items()):
        true_vals = results['true']
        pred_vals = results['pred']
        
        # 计算性能指标
        mae = np.mean(np.abs(np.array(pred_vals) - np.array(true_vals)))
        mse = np.mean((np.array(pred_vals) - np.array(true_vals)) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_vals, pred_vals)
        
        print(f"{sub} 回归性能:\n"
              f"  MAE: {mae:.2f}nM\n"
              f"  MSE: {mse:.2f}\n"
              f"  RMSE: {rmse:.2f}nM\n"
              f"  R²: {r2:.4f}")
        
        # 绘制散点图
        plt.subplot(1, len(substance_results), i+1)
        plt.scatter(true_vals, pred_vals, alpha=0.5)
        plt.plot([min(true_vals), max(true_vals)], 
                 [min(true_vals), max(true_vals)], 'r--', lw=2)
        plt.title(f'{sub} (MAE: {mae:.2f}nM)')
        plt.xlabel('真实浓度 (nM)')
        plt.ylabel('预测浓度 (nM)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'substance_regressions.png'))
    plt.close()
    
    return result_dir


def compare_models(model_paths, data_dir, backbone_names=None):
    """比较多个模型的性能"""
    if backbone_names is None:
        backbone_names = ['resnet18'] * len(model_paths)
    
    results = []
    
    # 分析每个模型
    for i, (model_path, backbone_name) in enumerate(zip(model_paths, backbone_names)):
        print(f"\n分析模型 {i+1}/{len(model_paths)}: {model_path}")
        result_dir = analyze_model(model_path, data_dir, backbone_name)
        
        # 获取模型名称
        model_name = os.path.basename(result_dir)
        results.append((model_name, result_dir))
    
    # 这里可以添加更多比较分析代码
    print("\n所有模型分析完成！")

if __name__ == '__main__':
    # 示例使用
    data_dir = 'data'
    model_path = 'results/resnet18_20250828_205702/best_model.pth'  # 替换为实际模型路径
    
    analyze_model(model_path, data_dir)