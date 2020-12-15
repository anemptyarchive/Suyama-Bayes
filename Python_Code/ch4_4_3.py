# ch4.4.3 ガウス混合モデルにおける変分推論

#%%

# 利用ライブラリ
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import psi
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
X_line, Y_line = np.meshgrid(
    np.arange(-10.0, 10.0, 0.2), np.arange(-10.0, 10.0, 0.2)
)
x_line = X_line.flatten()
y_line = Y_line.flatten()

# 観測モデルの確率密度を計算
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
plt.xlabel('$x_{n,1}$')
plt.ylabel('$x_{n,2}$')
plt.show()

#%%

# 事前分布のパラメータを指定
beta = 1
m_d = np.array([0.0, 0.0])
nu = D
w_dd = np.identity(D) * 0.05
alpha_k = np.repeat(1, K)

# 近似事後分布のパラメータの初期値をランダムに設定
beta_hat_k = np.random.choice(np.arange(0.1, 10.0, 0.1), size=K, replace=True)
m_hat_kd = np.random.choice(
    np.arange(np.min(np.round(x_nd, 1)), np.max(np.round(x_nd, 1)), 0.1), size=(K, D), replace=True
)
nu_hat_k = np.repeat(nu, K)
w_hat_kdd = np.array([w_dd for _ in range(K)])
alpha_hat_k = np.random.choice(np.arange(0.1, 10.0, 0.1), size=K, replace=True)

#%%

# 試行回数を指定
MaxIter = 250

# 途中計算に用いる項の受け皿を作成
ln_eta_nk = np.zeros((N, K))
E_lmd_kdd = np.zeros((K, D, D))
E_ln_det_lmd_k = np.zeros(K)
E_lmd_mu_kd = np.zeros((K, D))
E_mu_lmd_mu_k = np.zeros(K)
E_ln_pi_k = np.zeros(K)

# 推移の確認用の受け皿
trace_E_s_ink = np.zeros((MaxIter, N, K))
trace_E_mu_ikd = np.zeros((MaxIter+1, K, D))
trace_E_lambda_ikdd = np.zeros((MaxIter+1, K, D, D))
trace_E_mu_ikd[0] = m_hat_kd.copy()
trace_E_lambda_ikdd[0] = np.repeat(nu_hat_k, D * D).reshape(K, D, D) * w_hat_kdd

# 変分推論
for i in range(MaxIter):
    
    # Sの近似事後分布のパラメータを計算:式(4.109)
    for k in range(K):
        E_lmd_kdd[k] = nu_hat_k[k] * w_hat_kdd[k]
        E_ln_det_lmd_k[k] = np.sum(psi(0.5 * (nu_hat_k[k] - np.arange(D))))
        E_ln_det_lmd_k[k] += D * np.log(2) + np.log(np.linalg.det(w_hat_kdd[k]))
        E_lmd_mu_kd[k] = np.dot(E_lmd_kdd[k], m_hat_kd[k:k+1].T).flatten()
        E_mu_lmd_mu_k[k] = np.dot(m_hat_kd[k], E_lmd_mu_kd[k]) + D / beta_hat_k[k]
        E_ln_pi_k[k] = psi(alpha_hat_k[k]) - psi(np.sum(alpha_hat_k))
        ln_eta_nk[:, k] = -0.5 * np.diag(
            x_nd.dot(E_lmd_kdd[k]).dot(x_nd.T)
        )
        ln_eta_nk[:, k] += np.dot(x_nd, E_lmd_mu_kd[k:k+1].T).flatten()
        ln_eta_nk[:, k] -= 0.5 * E_mu_lmd_mu_k[k]
        ln_eta_nk[:, k] += 0.5 * E_ln_det_lmd_k[k] + E_ln_pi_k[k]
    tmp_eta_nk = np.exp(ln_eta_nk)
    eta_nk = ((tmp_eta_nk.T + 1e-7) / np.sum(tmp_eta_nk + 1e-7, axis=1)).T # 正規化
    
    # Sの近似事後分布の期待値を計算:式(4.59)
    E_s_nk = eta_nk.copy()
    
    for k in range(K):
        
        # muの近似事後分布のパラメータを計算:式(4.114)
        beta_hat_k[k] = np.sum(E_s_nk[:, k]) + beta
        m_hat_kd[k] = np.sum(E_s_nk[:, k] * x_nd.T, axis=1) + beta * m_d
        m_hat_kd[k] /= beta_hat_k[k]
        
        # lambdaの近似事後分布のパラメータを計算:式(4.118)
        nu_hat_k[k] = np.sum(E_s_nk[:, k]) + nu
        tmp_w_dd = np.dot(E_s_nk[:, k] * x_nd.T, x_nd)
        tmp_w_dd += beta * np.dot(m_d.reshape(D, 1), m_d.reshape(1, D))
        tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(D, 1), m_hat_kd[k].reshape(1, D))
        tmp_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
        
    # piの近似事後分布のパラメータを計算:式(4.58)
    alpha_hat_k = np.sum(E_s_nk, axis=0) + alpha_k
    
    # 観測モデルのパラメータの期待値を記録
    trace_E_s_ink[i] = E_s_nk.copy()
    trace_E_mu_ikd[i+1] = m_hat_kd.copy()
    trace_E_lambda_ikdd[i+1] = np.repeat(nu_hat_k, D * D).reshape(K, D, D) * w_hat_kdd
    
    # 動作確認
    print(str(i+1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

# 観測モデルを計算
z_kline = np.empty((K, len(x_line)))
for k in range(K):
    tmp_z_line = [
        multivariate_normal.pdf(
            x=(x, y), mean=m_hat_kd[k], cov=np.linalg.inv(nu_hat_k[k] * w_hat_kdd[k])
        ) for x, y in zip(x_line, y_line)
    ]
    z_kline[k] = tmp_z_line.copy()
Z_kline = z_kline.reshape((K, *X_line.shape))

#%%

# カラーマップを指定
cmap_list = ['Blues', 'Oranges', 'Greens']

# 各データのクラスタを抽出
max_p_idx = np.argmax(E_s_nk, axis=1)

# サンプルしたパラメータによるモデルを作図
fig = plt.figure(figsize=(10, 10))
for k in range(K):
    plt.contour(X_line, Y_line, Z_kline[k]) # 観測モデル
    plt.contour(X_line, Y_line, Z_true_kline[k], linestyles='dotted', alpha=0.5) # 真の観測モデル
    k_idx = np.where(max_p_idx == k) # クラスタkのインデックスを取得
    plt.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], c=E_s_nk[k_idx, k], cmap=cmap_list[k], label='cluster'+str(k+1)) # 観測データ
plt.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100, alpha=0.5) # 真の平均
plt.suptitle('Variational Inference', fontsize=20)
plt.title('K=' + str(K) + ', N=' + str(N) + ', iter:' + str(MaxIter), loc='left', fontsize=20)
plt.xlabel('$x_{n,1}$')
plt.ylabel('$x_{n,2}$')
plt.show()


#%%

# muの推移を確認
fig = plt.figure(figsize=(15, 10))
for k in range(K):
    for d in range(D):
        plt.plot(np.arange(MaxIter+1), trace_E_mu_ikd[:, k, d], 
                 label='k=' + str(k+1) + ', d=' + str(d+1))
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\mathbf{\mu}$' + ': N=' + str(N), loc='left', fontsize=20)
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
            plt.plot(np.arange(MaxIter+1), trace_E_lambda_ikdd[:, k, d1, d2], 
                 label='k=' + str(k+1) + ', d=' + str(d1+1) + ', d''=' + str(d2+1))
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\mathbf{\Lambda}$' + ': N=' + str(N), loc='left', fontsize=20)
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

# 描画する回数を指定(変更)
MaxIter = 100

# 観測モデルを計算
Z_ikline = np.zeros((MaxIter+1, K, *X_line.shape))
for i in range(MaxIter + 1):
    z_kline = np.empty((K, len(x_line)))
    for k in range(K):
        tmp_z_line = [
            multivariate_normal.pdf(
                x=(x, y), mean=trace_E_mu_ikd[i, k], cov=np.linalg.inv(trace_E_lambda_ikdd[i, k])
                ) for x, y in zip(x_line, y_line)
            ]
        z_kline[k] = tmp_z_line.copy()
    Z_ikline[i] = z_kline.reshape((K, *X_line.shape))
    
    # 動作確認
    print(str(i) + ' (' + str(np.round((i) / (MaxIter) * 100, 1)) + '%)')

#%%

# カラーマップを指定
cmap_list = ['Blues', 'Oranges', 'Greens']

# グラフを初期化
plt.cla()

# グラフを作成
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Variational Inference', fontsize=20)
ax = fig.add_subplot(1, 1, 1)

# 作図処理を関数として定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # 各データのクラスタを抽出
    max_p_idx = np.argmax(trace_E_s_ink[i], axis=1)
    
    # nフレーム目のグラフを描画
    for k in range(K):
        ax.contour(X_line, Y_line, Z_ikline[i, k]) # 観測モデル
        ax.contour(X_line, Y_line, Z_true_kline[k], linestyles='dotted', alpha=0.5) # 真の観測モデル
        if i > 0: # 初期値以外のとき
            k_idx = np.where(max_p_idx == k) # クラスタkのインデックスを取得
            ax.scatter(x_nd[k_idx, 0], x_nd[k_idx, 1], c=trace_E_s_ink[i, k_idx, k], 
                       cmap=cmap_list[k], label='cluster'+str(k+1)) # クラスタkの観測データ
    if i == 0: # 初期値のとき
        ax.scatter(x_nd[:, 0], x_nd[:, 1]) # 全ての観測データ
    ax.scatter(mu_true_kd[:, 0], mu_true_kd[:, 1], marker='+', s=100, alpha=0.5) # 真の平均
    
    # グラフの設定
    ax.set_title('K=' + str(K) + ', N=' + str(N) + ', iter:' + str(i), loc='left', fontsize=20)
    ax.set_xlabel('$x_{n,1}$')
    ax.set_ylabel('$x_{n,2}$')

# gif画像を作成
ani = animation.FuncAnimation(fig, update, frames=MaxIter + 1, interval=100)
ani.save("ch4_4_3_trace.gif")


#%%

print('end')

