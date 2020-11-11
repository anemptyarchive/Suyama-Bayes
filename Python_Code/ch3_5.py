# ch3.5 線形回帰の例

#%%

# 利用ライブラリ
import numpy as np
import matplotlib.pyplot as plt


# ch3.5.1 モデルの構築

#%%

# xのベクトル作成関数を定義
def x_vector(x_smp_n, M):
    x_mn = np.zeros(shape=(M, len(x_smp_n)))
    for m in range(M):
        x_mn[m, :] = np.power(x_smp_n, m)
    return x_mn

#%%

# 観測モデルのパラメータを指定
M_truth = 4
sigma = 1.0
lmd = 1.0 / sigma**2
w_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
).reshape([M_truth, -1])

#%%

# 作図用のx軸の値
x_line = np.arange(-3.0, 3.0, step=0.01)
y_line = np.dot(w_m.T, x_vector(x_line, M_truth)).flatten()

# ノイズを含まない観測モデルを作図
plt.plot(x_line, y_line)
plt.title('Observation Model', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# データをサンプリング
N = 50
x_smp_n = np.random.choice(
    np.arange(min(x_line), max(x_line), step=0.01), size=N, replace=True
)
y_1n = np.dot(w_m.T, x_vector(x_smp_n, M_truth))
y_1n += np.random.normal(loc=0.0, scale=1 / lmd, size=N) # ノイズ成分

#%%

# 観測データの散布図を作成
plt.scatter(x_smp_n, y_1n) # 観測データ
plt.plot(x_line, y_line) # 観測モデル
plt.title('Observation Model', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# 事前分布のパラメータを指定
M = 4
m_m = np.zeros((M, 1))
sigma_mm = np.identity(M)
lambda_mm = np.linalg.inv(sigma_mm**2)

# xのベクトル作成
x_mn = x_vector(x_smp_n, M)

#%%

# サンプリング数を指定
smp_size = 5

# 事前分布からwをサンプリング
x_mline = x_vector(x_line, M)
smp_model_arr = np.empty((smp_size, len(x_line)))
for i in range(smp_size):
    # wをサンプリング
    smp_w_m = np.random.multivariate_normal(
        mean=m_m.flatten(), cov=np.linalg.inv(lambda_mm), size=1
    ).reshape(M, -1)
    
    # 出力値を計算
    tmp_y_line = np.dot(smp_w_m.T, x_mline).flatten()
    smp_model_arr[i] = tmp_y_line.copy()

# 事前分布からサンプリングしたモデルを作図
for i in range(smp_size):
    plt.plot(x_line, smp_model_arr[i], label=str(i+1)) # 事前分布からサンプリングしたwによるモデル
plt.title('Sampling from Piror Distribution', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# 事後分布のパラメータを計算
lambda_hat_mm = lmd * np.dot(x_mn, x_mn.T) + lambda_mm
tmp_m_m = lmd * np.dot(x_mn, y_1n.T)
tmp_m_m += np.dot(lambda_mm, m_m)
m_hat_m = np.dot(np.linalg.inv(lambda_hat_mm), tmp_m_m)

#%%

# 事前分布からwをサンプリング
x_mline = x_vector(x_line, M)
smp_model_arr = np.empty((smp_size, len(x_line)))
for i in range(smp_size):
    # wをサンプリング
    smp_w_m = np.random.multivariate_normal(
        mean=m_hat_m.flatten(), cov=np.linalg.inv(lambda_hat_mm), size=1
    ).reshape(M, -1)
    
    # 出力値を計算
    tmp_y_line = np.dot(smp_w_m.T, x_mline).flatten()
    smp_model_arr[i] = tmp_y_line.copy()

# 事後分布からサンプリングしたモデルを作図
for i in range(smp_size):
    plt.plot(x_line, smp_model_arr[i], label=str(i+1)) # 事後分布からサンプリングしたwによるモデル
plt.scatter(x_smp_n, y_1n.flatten()) # 観測データ
plt.title('Sampling from Posterior Distribution', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# 予測分布のパラメータを計算
sigma2_star_hat_line = np.repeat(1 / lmd, len(x_line))
for i in range(len(x_line)):
    sigma2_star_hat_line[i] += x_mline[:, 1].T.dot(np.linalg.inv(lambda_hat_mm)).dot(x_mline[:, i])
mu_star_hat_line = np.dot(m_hat_m.T, x_mline).flatten()
print(sigma2_star_hat_line)
#%%

# 予測分布を作図
plt.plot(x_line, mu_star_hat_line, color='orange', label='predict') # 予測分布の期待値
plt.plot(x_line, mu_star_hat_line + np.sqrt(sigma2_star_hat_line), 
         color='#00A968', linestyle='--', label='$+\sigma$') # +sigma
plt.plot(x_line, mu_star_hat_line - np.sqrt(sigma2_star_hat_line), 
         color='#00A968', linestyle='--', label='$-\sigma$') # -sigma
plt.plot(x_line, y_line, linestyle=':', color='blue', label='model') # 観測モデル
plt.scatter(x_smp_n, y_1n, color='chocolate') # 観測データ
plt.title('Predict Distribution', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()

