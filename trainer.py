import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, cohen_kappa_score
import numpy as np
import os


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dl, val_dl, test_dl, device, logger, args, configs,
                 experiment_log_dir, seed):
        self.model = model  # 模型
        self.optimizer = optimizer  # 优化器
        self.scheduler = scheduler  # 学习率调度器
        self.train_dl = train_dl  # 训练数据加载器
        self.val_dl = val_dl  # 验证数据加载器
        self.test_dl = test_dl  # 测试数据加载器
        self.device = device  # 设备（CPU或GPU）
        self.logger = logger  # 日志记录器
        self.args = args  # 参数
        self.configs = configs  # 配置
        self.experiment_log_dir = experiment_log_dir  # 实验日志保存目录
        self.seed = seed  # 随机种子
        self.patience = configs.patience  # 早停耐心轮次
        self.best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
        self.early_stopping_counter = 0  # 早停计数器

    def train(self):
        for epoch in range(self.configs.epoch):
            self.model.train()  # 设置模型为训练模式
            total_loss = 0.0  # 总损失
            all_labels = []
            all_preds = []

            # 遍历训练数据集中的批次
            for batch in self.train_dl:
                # 将数据移动到设备上
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 清除梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)

                # 计算损失
                loss = self.compute_loss(outputs, labels)

                # 反向传播
                loss.backward()

                # 更新参数
                self.optimizer.step()

                # 累积总损失
                total_loss += loss.item()

                # 记录所有标签和预测
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

            # 计算本轮训练的平均损失
            avg_train_loss = total_loss / len(self.train_dl)
            train_mse = mean_squared_error(all_labels, all_preds)
            train_rmse = np.sqrt(train_mse)

            # 计算QWK
            all_preds_i = np.round(all_preds).astype(int)
            train_qwk = cohen_kappa_score(all_labels, all_preds_i, weights='quadratic')

            # 打印或记录本轮训练的平均损失、MSE和RMSE
            self.logger.info(
                f"第 {epoch + 1}/{self.configs.epoch} 轮，训练损失: {avg_train_loss}, MSE: {train_mse}, RMSE: {train_rmse}, QWK: {train_qwk}")

            # 验证模型
            val_loss = self.validate()

            # 检查早停条件
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # 保存当前最佳模型
                self.save_model()
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    self.logger.info(f"验证损失在 {self.patience} 轮内未改善，提前停止训练。")
                    break

            # 更新学习率调度器
            self.scheduler.step()

        # 保存预测结果和真实结果
        self.save_predictions(all_labels, all_preds)

    def validate(self):
        self.model.eval()  # 设置模型为评估模式
        total_val_loss = 0.0  # 总验证损失
        all_labels = []
        all_preds = []

        # 在验证过程中禁用梯度计算
        with torch.no_grad():
            for batch in self.val_dl:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)

                # 计算验证损失
                val_loss = self.compute_loss(outputs, labels)

                # 累积总验证损失
                total_val_loss += val_loss.item()

                # 记录所有标签和预测
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(self.val_dl)
        val_mse = mean_squared_error(all_labels, all_preds)
        val_rmse = np.sqrt(val_mse)

        all_preds_i = np.round(all_preds).astype(int)
        val_qwk = cohen_kappa_score(all_labels, all_preds_i, weights='quadratic')

        # 打印或记录平均验证损失、MSE和RMSE
        self.logger.info(f"验证损失: {avg_val_loss}, MSE: {val_mse}, RMSE: {val_rmse}, QWK: {val_qwk}")

        return avg_val_loss

    def test(self):
        self.model.eval()  # 设置模型为评估模式
        total_test_loss = 0.0  # 总测试损失
        all_labels = []
        all_preds = []

        # 在测试过程中禁用梯度计算
        with torch.no_grad():
            for batch in self.test_dl:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)

                # 计算测试损失
                test_loss = self.compute_loss(outputs, labels)

                # 累积总测试损失
                total_test_loss += test_loss.item()

                # 记录所有标签和预测
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        # 计算平均测试损失
        avg_test_loss = total_test_loss / len(self.test_dl)
        test_mse = mean_squared_error(all_labels, all_preds)
        test_rmse = np.sqrt(test_mse)

        all_preds_i = np.round(all_preds).astype(int)
        test_qwk = cohen_kappa_score(all_labels, all_preds_i, weights='quadratic')

        # 打印或记录平均测试损失、MSE和RMSE
        self.logger.info(f"测试损失: {avg_test_loss}, MSE: {test_mse}, RMSE: {test_rmse}, QWK: {test_qwk}")

    def compute_loss(self, outputs, labels):
        # 在这里定义损失函数
        labels = labels.float()  # 确保标签是浮点型
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, labels)
        return loss

    def save_predictions(self, labels, predictions):
        # 创建保存目录
        save_dir = os.path.join(self.experiment_log_dir, "predictions")
        os.makedirs(save_dir, exist_ok=True)

        # 将预测结果和真实结果保存到文件中
        save_path = os.path.join(save_dir, "results.csv")
        with open(save_path, 'w') as f:
            f.write("Real,Predicted\n")
            for real, pred in zip(labels, predictions):
                f.write(f"{real},{pred}\n")

        self.logger.info(f"预测结果已保存到 {save_path}")

    def save_model(self):
        # 保存模型的函数
        model_path = os.path.join(self.experiment_log_dir, "best_model.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"最佳模型已保存到 {model_path}")

    def compute_qwk(self, labels, preds):
        # 根据需要调整预测值和标签的格式
        # labels和preds应该是一维数组或列表
        qwk = cohen_kappa_score(labels, preds, weights='quadratic')
        return qwk
