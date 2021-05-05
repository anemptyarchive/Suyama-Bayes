# 4.4.3 ガウス混合モデルにおける推論：変分推論

#%%

# 4.4.3項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal # 多次元ガウス分布
from scipy.special import psi # ディガンマ関数
import matplotlib.pyplot as plt

#%%

## 観測モデルの設定

# 次元数を設定:(固定)
D = 2

# クラスタ数を指定
K = 3

# K個の真の平均を設定
mu_truth_kd = np.array(
    [[5.0, 35.0], 
     [-20.0, -10.0], 
     [30.0, -20.0]]
)

# K個の真の分散共分散行列を指定
sigma2_truth_kdd = np.array(
    [[[250.0, 65.0], [65.0, 270.0]], 
     [[125.0, -45.0], [-45.0, 175.0]], 
     [[210.0, -15.0], [-15.0, 250.0]]]
)
#lambda_truth_kdd = np.linalg.inv(sigma2_true_kdd)

# 真の混合比率を指定
pi_truth_k = np.array([0.45, 0.25, 0.3])

#%%

# 作図用のx軸の値を作成
x_1_line = np.linspace(
    np.min(mu_truth_kd[:, 0] - 3 * np.sqrt(sigma2_truth_kdd[:, 0, 0])), 
    np.max(mu_truth_kd[:, 0] + 3 * np.sqrt(sigma2_truth_kdd[:, 0, 0])), 
    num=300
)

# 作図用のy軸の値を作成
x_2_line = np.linspace(
    np.min(mu_truth_kd[:, 1] - 3 * np.sqrt(sigma2_truth_kdd[:, 1, 1])), 
    np.max(mu_truth_kd[:, 1] + 3 * np.sqrt(sigma2_truth_kdd[:, 1, 1])), 
    num=300
)

# 作図用の格子状の点を作成
x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)

# 作図用のxの点を作成
x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dim = x_1_grid.shape
print(x_dim)


# 観測モデルを計算
true_model = 0
for k in range(K):
    # クラスタkの確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=mu_truth_kd[k], cov=sigma2_truth_kdd[k]
    )
    
    # K個の確率密度の加重平均を計算
    true_model += tmp_density * pi_truth_k[k]

#%%

# 観測モデルを作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim)) # 真の分布
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.show()

#%%

# (観測)データ数を指定
N = 250

# クラスタを生成
s_truth_nk = np.random.multinomial(n=1, pvals=pi_truth_k, size=N)
print(s_truth_nk[:5])

# クラスタ番号を抽出
_, s_truth_n = np.where(s_truth_nk == 1)
print(s_truth_n[:5])

# (観測)データを生成
x_nd = np.array([
    np.random.multivariate_normal(
        mean=mu_truth_kd[k], cov=sigma2_truth_kdd[k], size=1
    ).flatten() for k in s_truth_n
])
print(x_nd[:5])

#%%

# 観測データの散布図を作成
plt.figure(figsize=(12, 9))
for k in range(K):
    k_idx, = np.where(s_truth_n == k)
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1)) # 観測データ
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), linestyles='--') # 真の分布
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('$N=' + str(N) + ', K=' + str(K) + 
          ', \pi=[' + ', '.join([str(pi) for pi in pi_truth_k]) + ']$', loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.show()

#%%

## 事前分布(ガウス・ウィシャート分布)の設定

# muの事前分布のパラメータを指定
beta = 1.0
m_d = np.repeat(0.0, D)

# lambdaの事前分布のパラメータを指定
nu = D
w_dd = np.identity(D) * 0.0005
print(np.sqrt(np.linalg.inv(beta * nu * w_dd))) # (疑似)相関行列

# piの事前分布のパラメータを指定
alpha_k = np.repeat(2.0, K)


# muの事前分布の標準偏差を計算
sigma_mu_d = np.sqrt(
    np.linalg.inv(beta * nu * w_dd)
).diagonal()

# 作図用のx軸の値
mu_0_line = np.linspace(
    np.min(m_d[0] - 3 * sigma_mu_d[0]), 
    np.max(m_d[0] + 3 * sigma_mu_d[0]), 
    num=300
)

# 作図用のy軸の値
mu_1_line = np.linspace(
    np.min(m_d[1] - 3 * sigma_mu_d[1]), 
    np.max(m_d[1] + 3 * sigma_mu_d[1]), 
    num=300
)

# 作図用の格子状の点を作成
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)

# 作図用のmuの点を作成
mu_point = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dim = mu_0_grid.shape
print(mu_dim)

#%%

# muの事前分布を計算
prior_density = multivariate_normal.pdf(
    x=mu_point, mean=m_d, cov=np.linalg.inv(beta * nu * w_dd)
)

# muの事前分布をを作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], color='red', s=100, marker='x') # 真の平均
plt.contour(mu_0_grid, mu_1_grid, prior_density.reshape(mu_dim)) # 事前分布
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.colorbar()
plt.show()

#%%

## 初期値の設定

# クラスタの初期値を生成
E_s_nk = np.random.uniform(low=0.0, high=1.0, size=(N, K))
E_s_nk /= np.sum(E_s_nk, axis=1, keepdims=True)
print(E_s_nk[:5])
print(np.sum(E_s_nk[:5], axis=1))

# 初期値によるmuの事後分布のパラメータを計算:式(4.114)
beta_hat_k = np.sum(E_s_nk, axis=0) + beta
m_hat_kd = (np.dot(E_s_nk.T, x_nd) + beta * m_d) / beta_hat_k.reshape((K, 1))
print(beta_hat_k)
print(m_hat_kd)

# 初期値によるlambdaの事後分布のパラメータを計算:式(4.118)
w_hat_kdd = np.zeros((K, D, D))
for k in range(K):
    inv_w_dd = np.dot(E_s_nk[:, k] * x_nd.T, x_nd)
    inv_w_dd += beta * np.dot(m_d.reshape((D, 1)), m_d.reshape((1, D)))
    inv_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape((D, 1)), m_hat_kd[k].reshape((1, D)))
    inv_w_dd += np.linalg.inv(w_dd)
    w_hat_kdd[k] = np.linalg.inv(inv_w_dd)
nu_hat_k = np.sum(E_s_nk, axis=0) + nu
print(w_hat_kdd)
print(nu_hat_k)

# 初期値によるpiの事後分布のパラメータを計算:式(4.58)
alpha_hat_k = np.sum(E_s_nk, axis=0) + alpha_k
print(alpha_hat_k)

#%%

# 初期値による混合分布を計算
init_density = 0
for k in range(K):
    # クラスタkの確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, 
        mean=m_hat_kd[k], 
        cov=np.linalg.inv(beta_hat_k[k] * nu_hat_k[k] * w_hat_kdd[k])
    )
    
    # K個の確率密度の加重平均を計算
    init_density += tmp_density * alpha_hat_k[k] / np.sum(alpha_hat_k)

# 初期値による分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), alpha=0.5, linestyles='dashed') # 真の分布
plt.contour(x_1_grid, x_2_grid, init_density.reshape(x_dim)) # 初期値による分布
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.show()

#%%

## 推論処理

# 試行回数を指定
MaxIter = 100

# 途中計算に用いる項の受け皿を作成
ln_eta_nk = np.zeros((N, K))

# 推移の確認用の受け皿
trace_E_s_ink = [E_s_nk.copy()]
trace_beta_ik = [beta_hat_k.copy()]
trace_m_ikd = [m_hat_kd.copy()]
trace_w_ikdd = [w_hat_kdd.copy()]
trace_nu_ik = [nu_hat_k.copy()]
trace_alpha_ik = [alpha_hat_k.copy()]

# 変分推論
for i in range(MaxIter):
    
    # 潜在変数の近似事後分布のパラメータを計算:式(4.109)
    for k in range(K):
        # クラスタkの中間変数を計算:式(4.119-4.122,4.62)
        E_lmd_dd = nu_hat_k[k] * w_hat_kdd[k]
        E_ln_det_lmd = np.sum(psi(0.5 * (nu_hat_k[k] - np.arange(D))))
        E_ln_det_lmd += D * np.log(2) + np.log(np.linalg.det(w_hat_kdd[k]))
        E_lmd_mu_d1 = np.dot(E_lmd_dd, m_hat_kd[k].reshape((D, 1)))
        E_mu_lmd_mu = np.dot(m_hat_kd[k].reshape((1, D)), E_lmd_mu_d1).item()
        E_mu_lmd_mu += D / beta_hat_k[k]
        E_ln_pi = psi(alpha_hat_k[k]) - psi(np.sum(alpha_hat_k))
        ln_eta_nk[:, k] = - 0.5 * np.diag(x_nd.dot(E_lmd_dd).dot(x_nd.T))
        ln_eta_nk[:, k] += np.dot(x_nd, E_lmd_mu_d1).flatten()
        ln_eta_nk[:, k] -= 0.5 * E_mu_lmd_mu + 0.5 * E_ln_det_lmd + E_ln_pi
    tmp_eta_nk = np.exp(ln_eta_nk)
    eta_nk = (tmp_eta_nk + 1e-7) / np.sum(tmp_eta_nk + 1e-7, axis=1, keepdims=True) # 正規化
    
    # 潜在変数の近似事後分布の期待値を計算:式(4.59)
    E_s_nk = eta_nk.copy()
    
    for k in range(K):
        
        # muの近似事後分布のパラメータを計算:式(4.114)
        beta_hat_k[k] = np.sum(E_s_nk[:, k]) + beta
        m_hat_kd[k] = (np.sum(E_s_nk[:, k] * x_nd.T, axis=1) + beta * m_d) / beta_hat_k[k]
        
        # lambdaの近似事後分布のパラメータを計算:式(4.118)
        inv_w_dd = np.dot(E_s_nk[:, k] * x_nd.T, x_nd)
        inv_w_dd += beta * np.dot(m_d.reshape((D, 1)), m_d.reshape((1, D)))
        inv_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape((D, 1)), m_hat_kd[k].reshape((1, D)))
        inv_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(inv_w_dd)
        nu_hat_k[k] = np.sum(E_s_nk[:, k]) + nu
        
    # piの近似事後分布のパラメータを計算:式(4.58)
    alpha_hat_k = np.sum(E_s_nk, axis=0) + alpha_k
    
    # i回目のパラメータを記録
    trace_E_s_ink.append(E_s_nk.copy())
    trace_beta_ik.append(beta_hat_k.copy())
    trace_m_ikd.append(m_hat_kd.copy())
    trace_w_ikdd.append(w_hat_kdd.copy())
    trace_nu_ik.append(nu_hat_k.copy())
    trace_alpha_ik.append(alpha_hat_k.copy())
    
    # 動作確認
    print(str(i+1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

## 推論結果の確認

# muの事後分布を計算
posterior_density_kg = np.empty((K, mu_point.shape[0]))
for k in range(K):
    # クラスタkのmuの事後分布を計算
    posterior_density_kg[k] = multivariate_normal.pdf(
        x=mu_point, 
        mean=m_hat_kd[k], 
        cov=np.linalg.inv(beta_hat_k[k] * nu_hat_k[k] * w_hat_kdd[k])
    )

# muの事後分布をを作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], color='red', s=100, marker='x') # 真の平均
for k in range(K):
    plt.contour(mu_0_grid, mu_1_grid, posterior_density_kg[k].reshape(mu_dim)) # 事後分布
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('iter:' + str(MaxIter) + ', N=' + str(N) + ', K=' + str(K), loc='left')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.colorbar()
plt.show()

#%%

# K個のカラーマップを指定
colormap_list = ['Blues', 'Oranges', 'Greens']

# 確率が最大のクラスタ番号を抽出
s_n = np.argmax(E_s_nk, axis=1)

# 各データのクラスタとなる確率を抽出
prob_s_n = E_s_nk[np.arange(N), s_n]
print(prob_s_n[:5])

# 最後のサンプルによる混合分布を計算
res_density = 0
for k in range(K):
    # クラスタkの確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, 
        mean=m_hat_kd[k], 
        cov=np.linalg.inv(nu_hat_k[k] * w_hat_kdd[k])
    )
    
    # K個の確率密度の加重平均を計算
    res_density += tmp_density * alpha_hat_k[k] / np.sum(alpha_hat_k)

# 最後のサンプルによる分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), alpha=0.5, linestyles='dashed') # 真の分布
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], color='red', s=100, marker='x') # 真の平均
plt.contour(x_1_grid, x_2_grid, res_density.reshape(x_dim)) # 最後のサンプルによる分布:(等高線)
#plt.contourf(x_1_grid, x_2_grid, res_density.reshape(x_dim), alpha=0.5) # 最後のサンプルによる分布:(塗りつぶし)
for k in range(K):
    k_idx, = np.where(s_n == k)
    cm = plt.get_cmap(colormap_list[k]) # クラスタkのカラーマップを設定
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1), 
                c=[cm(p) for p in prob_s_n[k_idx]]) # サンプルしたクラスタ
plt.suptitle('Gaussian Mixture Model:Variational Inference', fontsize=20)
plt.title('iter:' + str(MaxIter) + ', N=' + str(N) + ', K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.legend()
plt.show()

#%%

## 超パラメータの更新値の推移の確認

# betaの推移を確認
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_beta_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\hat{\\beta}$', loc='left')
plt.legend()
plt.show()

#%%

# mの推移を確認
plt.figure(figsize=(12, 9))
for k in range(K):
    for d in range(D):
        plt.plot(np.arange(MaxIter+1), np.array(trace_m_ikd)[:, k, d], 
                 label='k=' + str(k + 1) + ', d=' + str(d + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\hat{m}$', loc='left')
plt.legend()
plt.show()

#%%

# wの推移を確認
plt.figure(figsize=(12, 9))
for k in range(K):
    for d1 in range(D):
        for d2 in range(D):
            plt.plot(np.arange(MaxIter + 1), np.array(trace_w_ikdd)[:, k, d1, d2], 
                     alpha=0.5, label='k=' + str(k + 1) + ', d=' + str(d1 + 1) + ', d''=' + str(d2 + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\hat{w}$', loc='left')
plt.legend()
plt.show()

#%%

# nuの推移を確認
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_nu_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\hat{\\nu}$', loc='left')
plt.legend()
plt.show()

#%%

# alphaの推移を確認
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_alpha_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\hat{\\alpha}$', loc='left')
plt.legend()
plt.show()

#%%

## アニメーションによる確認

# 追加ライブラリ
import matplotlib.animation as animation

#%%

## 事後分布の推移の確認

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_posterior(i):
    # i回目のmuの事後分布を計算
    posterior_density_kg = np.empty((K, mu_point.shape[0]))
    for k in range(K):
        # クラスタkのmuの事後分布を計算
        posterior_density_kg[k] = multivariate_normal.pdf(
            x=mu_point, 
            mean=trace_m_ikd[i][k], 
            cov=np.linalg.inv(trace_beta_ik[i][k] * trace_nu_ik[i][k] * trace_w_ikdd[i][k])
        )
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のmuの事後分布をを作図
    plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], color='red', s=100, marker='x') # 真の平均
    for k in range(K):
        plt.contour(mu_0_grid, mu_1_grid, posterior_density_kg[k].reshape(mu_dim)) # 事後分布
    plt.suptitle('Gaussian Distribution:Variational Inference', fontsize=20)
    plt.title('iter:' + str(i) + ', N=' + str(N) + ', K=' + str(K), loc='left')
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=MaxIter + 1, interval=100)
posterior_anime.save("ch4_4_3_Posterior.gif")

#%%

## サンプルによる分布の推移の確認

# K個のカラーマップを指定
colormap_list = ['Blues', 'Oranges', 'Greens']

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_model(i):
    # i回目のサンプルによる混合分布を計算
    res_density = 0
    for k in range(K):
        # クラスタkの確率密度を計算
        tmp_density = multivariate_normal.pdf(
            x=x_point, 
            mean=trace_m_ikd[i][k], 
            cov=np.linalg.inv(trace_nu_ik[i][k] * trace_w_ikdd[i][k])
        )
        
        # K個の確率密度の加重平均を計算
        res_density += tmp_density * trace_alpha_ik[i][k] / np.sum(trace_alpha_ik[i])
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のサンプルによる分布を作図
    plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), alpha=0.5, linestyles='dashed') # 真の分布
    plt.contourf(x_1_grid, x_2_grid, res_density.reshape(x_dim), alpha=0.5) # 推定した分布
    #plt.contour(x_1_grid, x_2_grid, res_density.reshape(x_dim)) # 推定した分布
    plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], color='red', s=100, marker='x') # 真の平均
    for k in range(K):
        k_idx, = np.where(np.argmax(trace_E_s_ink[i], axis=1) == k)
        cm = plt.get_cmap(colormap_list[k]) # クラスタkのカラーマップを設定
        plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1), 
                    c=[cm(p) for p in trace_E_s_ink[i][k_idx, k]]) # クラスタのサンプル
        plt.suptitle('Gaussian Mixture Model:Variational Inference', fontsize=20)
    plt.title('iter:' + str(i) + ', N=' + str(N) + ', K=' + str(K), loc='left')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

# gif画像を作成
model_anime = animation.FuncAnimation(fig, update_model, frames=MaxIter + 1, interval=100)
model_anime.save("ch4_4_3_Model.gif")


#%%

print('end')

