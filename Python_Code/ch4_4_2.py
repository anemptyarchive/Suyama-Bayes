# ch4.4.2 ガウス混合モデルにおけるギブスサンプリング

#%%

# 利用ライブラリ
import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet
import matplotlib.pyplot as plt


# モデルの設定

#%%

# 真の観測モデルのパラメータを指定
D = 2
K = 3
mu_true_kd = np.array([
    [0.0, 4.0], 
     [-5.0, -5.0], 
     [5.0, -2.5]
])
sigma2_true_kdd = np.array([
    [[8.0, 0.0], [0.0, 8.0]], 
    [[4.0, -2.5], [-2.5, 4.0]], 
    [[6.5, 4.0], [4.0, 6.5]]
])
lambda_true_kdd = np.linalg.inv(sigma2_true_kdd)

# 真の混合比率を指定
pi_true_k = np.array([0.5, 0.2, 0.3])

#%%

# 各データの真のクラスタを生成
N = 250
s_true_nk = np.random.multinomial(n=1, pvals=pi_true_k, size=N)
_, s_true_n = np.where(s_true_nk == 1)

# 観測データを生成
x_nd = np.array([
    np.random.multivariate_normal(
        mean=mu_true_kd[k], cov=sigma2_true_kdd[k], size=1
    ).flatten() for k in s_true_n
])

#%%

# 作図用の格子状の点を作成
X_line, Y_line = np.meshgrid(np.arange(-10.0, 10.0, 0.1), np.arange(-10.0, 10.0, 0.1))
x_line = X_line.flatten()
y_line = Y_line.flatten()

# 観測モデルを計算
z_kline = np.empty((K, len(x_line)))
for k in range(K):
    tmp_z_line = [
        multivariate_normal.pdf(
            (x, y), mean=mu_true_kd[k], cov=sigma2_true_kdd[k]
        ) for x, y in zip(x_line, y_line)
    ]
    z_kline[k] = tmp_z_line.copy()
Z_true_kline = z_kline.reshape((K, *X_line.shape))

#%%

# 観測データの散布図を作成
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    plt.contour(X_line, Y_line, Z_true_kline[k]) # 真の観測モデル
    k_idx = np.where(s_true_n == k)
    plt.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], label='cluster'+str(k+1)) # 真の観測データ
plt.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100) # 真の平均
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('K=' + str(K) + ', N=' + str(N), loc='left', fontsize=20)
plt.show()

#%%

# 観測モデルのパラメータの初期値を指定
mu_kd = np.zeros((K, D))
lambda_kdd = np.array([
    np.linalg.inv(np.identity(D) * 10) for _ in range(K)
])

# 混合比率の初期値を指定
pi_k = np.random.choice(np.arange(0.0, 1.0, 0.01), size=K)
pi_k /= np.sum(pi_k)
alpha_k = np.repeat(1, K)

# 事前分布のパラメータを指定
beta = 1
m_d = np.array([0.0, 0.0])
sigma_dd = np.identity(D) * 10
lambda_kdd = np.array([
    np.linalg.inv(sigma_dd**2) for _ in range(K)
])
nu = D
w_dd = np.identity(D) * 10

print(pi_k)

#%%

# 試行回数を指定
MaxIter = 3000

# ギブスサンプリング
for i in range(MaxIter):
    
    # 初期化
    eta_nk = np.zeros((N, K))
    s_nk = np.zeros((N, K))
    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, D))
    nu_hat_k = np.zeros(K)
    w_hat_kdd = np.zeros((K, D, D))
    alpha_hat_k = np.zeros(K)
    
    # 潜在変数のパラメータを計算：式(4.94)
    for k in range(K):
        tmp_eta_nn = -0.5 * (x_nd - mu_kd[k]).dot(lambda_kdd[k]).dot((x_nd - mu_kd[k]).T)
        tmp_eta_nn += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
        tmp_eta_nn += np.log(pi_k[k] + 1e-7)
        eta_nk[:, k] = np.exp(np.diag(tmp_eta_nn))
    eta_nk /= np.sum(eta_nk, axis=1, keepdims=True) # 正規化
    
    # 潜在変数をサンプル：式(4.93)
    for n in range(N):
        s_nk[n] = np.random.multinomial(n=1, pvals=eta_nk[n], size=1).flatten()
    
    # 観測モデルのパラメータをサンプリング
    for k in range(K):
        
        # muの事後分布のパラメータを計算：式(4.99)
        beta_hat_k[k] = np.sum(s_nk[:, k]) + beta
        m_hat_kd[k] = np.sum(s_nk[:, k] * x_nd.T, axis=1)
        m_hat_kd[k] += beta * m_d
        m_hat_kd[k] /= beta_hat_k[k]
        
        # lambdaの事後分布のパラメータを計算：式(4.103)
        nu_hat_k[k] = np.sum(s_nk[:, k]) + nu
        tmp_w_dd = np.dot((s_nk[:, k] * x_nd.T), x_nd)
        tmp_w_dd += beta * np.dot(m_d.reshape(D, 1), m_d.reshape(1, D))
        tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(D, 1), m_hat_kd[k].reshape(1, D))
        tmp_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
        
        # lambdaをサンプル：式(4.102)
        lmd_sampler = wishart(df=nu_hat_k[k], scale=w_hat_kdd[k])
        lambda_kdd[k] = lmd_sampler.rvs(size=1)
        
        # muをサンプル：式(4.98)
        mu_kd[k] = np.random.multivariate_normal(
            mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1
        ).flatten()
    
    # 混合比率のパラメータを計算：式(4.45)
    alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
    
    # piをサンプル：式(4.44)
    pi_sampler = dirichlet(alpha=alpha_hat_k)
    pi_k = pi_sampler.rvs(size=1).flatten()
    
    # 動作確認
    print(str(i+1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

# 観測モデルを計算
z_kline = np.empty((K, len(x_line)))
for k in range(K):
    tmp_z_line = [
        multivariate_normal.pdf(
            (x, y), mean=mu_kd[k], cov=np.linalg.inv(lambda_kdd[k])
        ) for x, y in zip(x_line, y_line)
    ]
    z_kline[k] = tmp_z_line.copy()
Z_kline = z_kline.reshape((K, *X_line.shape))

#%%

# 各データのクラスタを抽出
_, s_n = np.where(s_nk == 1)

# 観測データの散布図を作成
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    plt.contour(X_line, Y_line, Z_kline[k]) # 観測モデル
    plt.contour(X_line, Y_line, Z_true_kline[k], linestyles='dotted', alpha=0.5) # 真の観測モデル
    k_idx = np.where(s_n == k) # クラスタkのインデックスを取得
    plt.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], label='cluster'+str(k+1)) # 観測データ
plt.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100, alpha=0.5) # 真の平均
plt.suptitle('Gibbs Sampling', fontsize=20)
plt.title('K=' + str(K) + ', N=' + str(N), loc='left', fontsize=20)
plt.show()

#%%

print('end')

