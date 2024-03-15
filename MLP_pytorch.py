import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score



# 读取CSV文件
data = pd.read_csv('input_remove.csv')

X = data[['time', 'Distance (um)', 'Distance to position', 'Average Content of C (wt. %)', 'Average Content of N (wt. %)', 'Theoretical value of cementite']].values
y = data['Actual value of cementite'].values

# 归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.layer2 = nn.Linear(32, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = MLP(X_train.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)

num_epochs = 500
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss_train = criterion(outputs, y_train)
    loss_train.backward()
    optimizer.step()

    # Evaluate on test data
    model.eval()
    # 计算R^2和MSE
    with torch.no_grad():
        y_pred_test = model(X_test).numpy()
        y_actual_test = y_test.numpy()
        r2 = r2_score(y_actual_test, y_pred_test)
        mse = np.mean((y_pred_test - y_actual_test) ** 2)

        y_pred_train = model(X_train).numpy()
        y_actual_train = y_train.numpy()
        loss_test = criterion(model(X_test), y_test)

    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())

    # 使用验证损失更新调度器
    scheduler.step(loss_test)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_train.item():.4e}, Test Loss: {loss_test.item():.4e}')

    plt.figure(figsize=(10, 10))
    plt.scatter(y_actual_train, y_pred_train, color='blue', label="Training Data")
    plt.scatter(y_actual_test, y_pred_test, color='green', label="Test Data")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=3, color='red')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"Actual vs. Predicted Value\nR2 = {r2:.4f}, MSE = {mse:.4e}")
    plt.legend()
    plt.grid(True)
    plt.savefig('./pic_remove/Actual vs. Predicted Value_1.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", lw=2)
    plt.plot(test_losses, label="Validation Loss", lw=2)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig('./pic_remove/Loss vs. Epoch.png')
    plt.show()

with torch.no_grad():
    y_pred_train = model(X_train).numpy()
    y_actual_train = y_train.numpy()

plt.figure(figsize=(10, 10))
plt.scatter(y_actual_train, y_pred_train, color='blue', label="Training Data")
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=3, color='red')
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs. Predicted Value")
plt.legend()
plt.grid(True)
plt.savefig('./pic_remove/Actual vs. Predicted Value_2.png')
plt.show()


torch.save(model.state_dict(), './model_1.pth')


# 计算相关矩阵
correlation_matrix = data[['time', 'Distance (um)', 'Distance to position', 'Average Content of C (wt. %)', 'Average Content of N (wt. %)', 'Theoretical value of cementite', 'Actual value of cementite']].corr()


# 创建一个遮盖矩阵，只遮盖上半部分，保留对角线
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# 使用掩码绘制热力图
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu',
            cbar_kws={'label': 'Correlation coefficient'},
            mask=mask, square=True, vmin=-1, vmax=1)  # 设置颜色范围以匹配之前的文献中的图像


# 计算权重，使直方图的y轴表示占比
weights = np.ones_like(correlation_matrix.values.flatten()) / len(correlation_matrix.values.flatten())

# 3. 在热力图右侧绘制直方图
ax_hist = plt.axes([0.53, 0.53, 0.25, 0.25])
ax_hist.hist(correlation_matrix.values.flatten(), bins=15, weights=weights, color='#337eb8',edgecolor='white')  # 减少bins数量以使柱子更粗
ax_hist.set_title('Distribution of Pearson Coefficients')
ax_hist.set_xlabel('Pearson Coefficients')
ax_hist.set_ylabel('Proportion')
plt.savefig('./pic_remove/Correlation Matrix.png')
plt.show()





model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    loss = criterion(y_pred, y_test)
print(f'Test Loss: {loss.item():.4e}')
