# ch4.4.2 ガウス混合モデルにおけるギブスサンプリング

#%%

# 利用ライブラリ
import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet
import matplotlib.pyplot as plt

#%%

# モデルの設定

#%%

# 真の観測モデルのパラメータを指定
D = 2
K = 3
mu_true_kd = np.array(
    [[0.0, 4.0], 
     [-5.0, -5.0], 
     [5.0, -2.5]]
)
sigma2_true_kdd = np.array(
    [[[8.0, 0.0], [0.0, 8.0]], 
     [[4.0, -2.5], [-2.5, 4.0]], 
     [[6.5, 4.0], [4.0, 6.5]]]
)
#lambda_true_kdd = np.linalg.inv(sigma2_true_kdd)

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
X_line, Y_line = np.meshgrid(np.arange(-10.0, 10.0, 0.2), np.arange(-10.0, 10.0, 0.2))
x_line = X_line.flatten()
y_line = Y_line.flatten()

# 観測モデルを計算
z_kline = np.empty((K, len(x_line)))
for k in range(K):
    tmp_z_line = [
        multivariate_normal.pdf(
            x=(x, y), mean=mu_true_kd[k], cov=sigma2_true_kdd[k]
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
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

#%%

# 観測モデルのパラメータの初期値を指定
mu_kd = np.zeros((K, D))
lambda_kdd = np.array([
    np.linalg.inv(np.identity(D) * 100) for _ in range(K)
])

# 混合比率の初期値を指定
pi_k = np.random.choice(np.arange(0.0, 1.0, 0.01), size=K)
pi_k /= np.sum(pi_k)

# 事前分布のパラメータを指定
beta = 1
m_d = np.array([0.0, 0.0])
nu = D
w_dd = np.identity(D) * 10
alpha_k = np.repeat(1, K)

print(pi_k)

#%%

# 試行回数を指定
MaxIter = 250

# 推移の確認用の受け皿
trace_s_in = np.zeros((MaxIter, N))
trace_mu_ikd = np.zeros((MaxIter+1, K, D))
trace_lambda_ikdd = np.zeros((MaxIter+1, K, D, D))
trace_mu_ikd[0] = mu_kd.copy()
trace_lambda_ikdd[0] = lambda_kdd.copy()

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
        tmp_eta_n = np.diag(
            -0.5 * (x_nd - mu_kd[k]).dot(lambda_kdd[k]).dot((x_nd - mu_kd[k]).T)
        ).copy()
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
        tmp_eta_n += np.log(pi_k[k] + 1e-7)
        eta_nk[:, k] = np.exp(tmp_eta_n)
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
        lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
        
        # muをサンプル：式(4.98)
        mu_kd[k] = np.random.multivariate_normal(
            mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1
        ).flatten()
    
    # 混合比率のパラメータを計算：式(4.45)
    alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
    
    # piをサンプル：式(4.44)
    pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
    
    # 値を記録
    _, trace_s_in[i] = np.where(s_nk == 1)
    trace_mu_ikd[i+1] = mu_kd.copy()
    trace_lambda_ikdd[i+1] = lambda_kdd.copy()
    
    # 動作確認
    print(str(i+1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

# 観測モデルを計算
z_kline = np.empty((K, len(x_line)))
for k in range(K):
    tmp_z_line = [
        multivariate_normal.pdf(
            x=(x, y), mean=mu_kd[k], cov=np.linalg.inv(lambda_kdd[k])
        ) for x, y in zip(x_line, y_line)
    ]
    z_kline[k] = tmp_z_line.copy()
Z_kline = z_kline.reshape((K, *X_line.shape))

#%%

# 各データのクラスタを抽出
_, s_n = np.where(s_nk == 1)

# サンプルしたパラメータによるモデルを作図
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    plt.contour(X_line, Y_line, Z_kline[k]) # 観測モデル
    plt.contour(X_line, Y_line, Z_true_kline[k], linestyles='dotted', alpha=0.5) # 真の観測モデル
    k_idx = np.where(s_n == k) # クラスタkのインデックスを取得
    plt.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], label='cluster'+str(k+1)) # 観測データ
plt.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100, alpha=0.5) # 真の平均
plt.suptitle('Gibbs Sampling', fontsize=20)
plt.title('K=' + str(K) + ', N=' + str(N) + ', iter:' + str(MaxIter), loc='left', fontsize=20)
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.show()

#%%

# muの推移を確認
fig = plt.figure(figsize=(15, 10))
for k in range(K):
    for d in range(D):
        plt.plot(np.arange(MaxIter+1), trace_mu_ikd[:, k, d], 
                 label='k=' + str(k+1) + ', d=' + str(d+1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend()
plt.show()

#%%

# lambdaの推移を確認
fig = plt.figure(figsize=(15, 10))
for k in range(K):
    for d1 in range(D):
        for d2 in range(D):
            plt.plot(np.arange(MaxIter+1), trace_lambda_ikdd[:, k, d1, d2], 
                 label='k=' + str(k+1) + ', d=' + str(d1+1) + ', d''=' + str(d2+1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend()
plt.show()

#%%

# gif画像で推移の確認

#%%

# 追加ライブラリ
import matplotlib.animation as animation

#%%

# 観測モデルを計算
Z_ikline = np.zeros((MaxIter+1, K, *X_line.shape))
for i in range(MaxIter + 1):
    z_kline = np.empty((K, len(x_line)))
    for k in range(K):
        tmp_z_line = [
            multivariate_normal.pdf(
                x=(x, y), mean=trace_mu_ikd[i, k], cov=np.linalg.inv(trace_lambda_ikdd[i, k])
                ) for x, y in zip(x_line, y_line)
            ]
        z_kline[k] = tmp_z_line.copy()
    Z_ikline[i] = z_kline.reshape((K, *X_line.shape))
    
    # 動作確認
    print(str(i) + ' (' + str(np.round((i) / (MaxIter) * 100, 1)) + '%)')

#%%

# グラフを初期化
plt.cla()

# グラフを作成
fig = plt.figure(figsize=(12, 12))
fig.suptitle('Gibbs Sampling', fontsize=20)
ax = fig.add_subplot(1, 1, 1)

# 作図処理を関数として定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # nフレーム目のグラフを描画
    for k in range(K):
        ax.contour(X_line, Y_line, Z_ikline[i, k]) # 観測モデル
        ax.contour(X_line, Y_line, Z_true_kline[k], linestyles='dotted', alpha=0.5) # 真の観測モデル
        if i > 0: # 初期値以外のとき
            k_idx = np.where(trace_s_in[i-1] == k) # クラスタkのインデックスを取得
            ax.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], label='cluster'+str(k+1)) # クラスタkの観測データ
    if i == 0: # 初期値のとき
        ax.scatter(x_nd[:, 0], x_nd[:, 1]) # 全ての観測データ
    ax.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100, alpha=0.5) # 真の平均
    
    # グラフの設定
    ax.set_title('K=' + str(K) + ', N=' + str(N) + ', iter:' + str(i), loc='left', fontsize=20)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')

# gif画像を作成
ani = animation.FuncAnimation(fig, update, frames=MaxIter + 1, interval=100)
#ani.save("ch4_4_2_trace.gif")


#%%

# burn-in？

'''
未完というか不明
'''

#%%

# 事後分布のパラメータを事前分布のパラメータに設定
beta_k = beta_hat_k.copy()
m_kd = m_hat_kd.copy()
nu_k = nu_hat_k.copy()
w_kdd = w_hat_kdd.copy()
alpha_k = alpha_hat_k.copy()

# 試行回数を指定
MaxIter = 300

# 推移の確認用の受け皿
trace_s_in = np.zeros((MaxIter, N))
trace_mu_ikd = np.zeros((MaxIter+1, K, D))
trace_lambda_ikdd = np.zeros((MaxIter+1, K, D, D))
trace_mu_ikd[0] = mu_kd.copy()
trace_lambda_ikdd[0] = lambda_kdd.copy()

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
        tmp_eta_n = np.diag(
            -0.5 * (x_nd - mu_kd[k]).dot(lambda_kdd[k]).dot((x_nd - mu_kd[k]).T)
        ).copy()
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
        tmp_eta_n += np.log(pi_k[k] + 1e-7)
        eta_nk[:, k] = np.exp(tmp_eta_n)
    eta_nk /= np.sum(eta_nk, axis=1, keepdims=True) # 正規化
    
    # 潜在変数をサンプル：式(4.93)
    for n in range(N):
        s_nk[n] = np.random.multinomial(n=1, pvals=eta_nk[n], size=1).flatten()
    
    # 観測モデルのパラメータをサンプリング
    for k in range(K):
        
        # muの事後分布のパラメータを計算：式(4.99)
        beta_hat_k[k] = np.sum(s_nk[:, k]) + beta_k[k]
        m_hat_kd[k] = np.sum(s_nk[:, k] * x_nd.T, axis=1)
        m_hat_kd[k] += beta_k[k] * m_kd[k]
        m_hat_kd[k] /= beta_hat_k[k]
        
        # lambdaの事後分布のパラメータを計算：式(4.103)
        nu_hat_k[k] = np.sum(s_nk[:, k]) + nu_k[k]
        tmp_w_dd = np.dot((s_nk[:, k] * x_nd.T), x_nd)
        tmp_w_dd += beta_k[k] * np.dot(m_kd[k].reshape(D, 1), m_kd[k].reshape(1, D))
        tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(D, 1), m_hat_kd[k].reshape(1, D))
        tmp_w_dd += np.linalg.inv(w_kdd[k])
        w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
        
        # lambdaをサンプル：式(4.102)
        lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
        
        # muをサンプル：式(4.98)
        mu_kd[k] = np.random.multivariate_normal(
            mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1
        ).flatten()
    
    # 混合比率のパラメータを計算：式(4.45)
    alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
    
    # piをサンプル：式(4.44)
    pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
    
    # 値を記録
    _, trace_s_in[i] = np.where(s_nk == 1)
    trace_mu_ikd[i+1] = mu_kd.copy()
    trace_lambda_ikdd[i+1] = lambda_kdd.copy()
    
    # 動作確認
    print(str(i+1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

# 観測モデルを計算
z_kline = np.empty((K, len(x_line)))
for k in range(K):
    tmp_z_line = [
        multivariate_normal.pdf(
            x=(x, y), mean=mu_kd[k], cov=np.linalg.inv(lambda_kdd[k])
        ) for x, y in zip(x_line, y_line)
    ]
    z_kline[k] = tmp_z_line.copy()
Z_kline = z_kline.reshape((K, *X_line.shape))

#%%

# 各データのクラスタを抽出
_, s_n = np.where(s_nk == 1)

# サンプルしたパラメータによるモデルを作図
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    plt.contour(X_line, Y_line, Z_kline[k]) # 観測モデル
    plt.contour(X_line, Y_line, Z_true_kline[k], linestyles='dotted', alpha=0.5) # 真の観測モデル
    k_idx = np.where(s_n == k) # クラスタkのインデックスを取得
    plt.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], label='cluster'+str(k+1)) # 観測データ
plt.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100, alpha=0.5) # 真の平均
plt.suptitle('Gibbs Sampling', fontsize=20)
plt.title('K=' + str(K) + ', N=' + str(N) + ', iter:' + str(MaxIter), loc='left', fontsize=20)
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.show()

#%%

# muの推移を確認
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    for d in range(D):
        plt.plot(np.arange(MaxIter+1), trace_mu_ikd[:, k, d], 
                 label='k=' + str(k+1) + ', d=' + str(d+1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend()
plt.show()

#%%

# lambdaの推移を確認
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    for d1 in range(D):
        for d2 in range(D):
            plt.plot(np.arange(MaxIter+1), trace_lambda_ikdd[:, k, d1, d2], 
                 label='k=' + str(k+1) + ', d=' + str(d1+1) + ', d''=' + str(d2+1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend()
plt.show()

#%%

print('end')

