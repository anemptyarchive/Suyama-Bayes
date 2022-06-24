# 4.4.4 ガウス混合モデルにおける推論：崩壊型ギブスサンプリング

#%%

# 4.4.4項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t # 多次元ガウス分布, 多次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 観測モデルの設定

# 次元数を設定:(固定)
D = 2

# クラスタ数を指定
K = 3

# K個の真の平均を指定
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
#lambda_truth_kdd = np.linalg.inv(sigma2_truth_kdd)

# 真の混合比率を指定
pi_truth_k = np.array([0.45, 0.25, 0.3])

#%%

# 作図用のx軸のxの値を作成
x_1_line = np.linspace(
    np.min(mu_truth_kd[:, 0] - 3 * np.sqrt(sigma2_truth_kdd[:, 0, 0])), 
    np.max(mu_truth_kd[:, 0] + 3 * np.sqrt(sigma2_truth_kdd[:, 0, 0])), 
    num=300
)

# 作図用のy軸のxの値を作成
x_2_line = np.linspace(
    np.min(mu_truth_kd[:, 1] - 3 * np.sqrt(sigma2_truth_kdd[:, 1, 1])), 
    np.max(mu_truth_kd[:, 1] + 3 * np.sqrt(sigma2_truth_kdd[:, 1, 1])), 
    num=300
)

# 作図用の格子状の点を作成
x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)

# 作図用のxの点を作成
x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)

# 作図用に各次元の要素数を保存
x_dim = x_1_grid.shape
print(x_dim)


# 観測モデルを計算
true_model = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=mu_truth_kd[k], cov=sigma2_truth_kdd[k]
    )
    
    # K個の分布の加重平均を計算
    true_model += pi_truth_k[k] * tmp_density

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

# 潜在変数を生成
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
    k_idx, = np.where(s_truth_n == k) # クラスタkのデータのインデック
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
w_dd = np.identity(D) * 0.0005
nu = D
print(np.sqrt(np.linalg.inv(beta * nu * w_dd))) # (疑似)相関行列

# piの事前分布のパラメータを指定
alpha_k = np.repeat(2.0, K)


# muの事前分布の標準偏差を計算
sigma_mu_d = np.sqrt(
    np.linalg.inv(beta * nu * w_dd).diagonal()
)

# 作図用のx軸のmuの値を作成
mu_0_line = np.linspace(
    np.min(mu_truth_kd[:, 0]) - sigma_mu_d[0], 
    np.max(mu_truth_kd[:, 0]) + sigma_mu_d[0], 
    num=300
)

# 作図用のy軸のmuの値を作成
mu_1_line = np.linspace(
    np.min(mu_truth_kd[:, 1]) - sigma_mu_d[1], 
    np.max(mu_truth_kd[:, 1]) + sigma_mu_d[1], 
    num=300
)

# 作図用の格子状の点を作成
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)

# 作図用のmuの点を作成
mu_point = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)

# 作図用に各次元の要素数を保存
mu_dim = mu_0_grid.shape
print(mu_dim)

#%%

# muの事前分布を計算
prior_mu_density = multivariate_normal.pdf(
    x=mu_point, mean=m_d, cov=np.linalg.inv(beta * nu * w_dd)
)

# muの事前分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], label='true val', 
            color='red', s=100, marker='x') # 真の平均
plt.contour(mu_0_grid, mu_1_grid, prior_mu_density.reshape(mu_dim)) # 事前分布
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.colorbar()
plt.legend()
plt.show()

#%%

## 初期値の設定

# 潜在変数を初期化
s_nk = np.random.multinomial(n=1, pvals=alpha_k / np.sum(alpha_k), size=N)
print(s_nk[:5])

# 初期値によるmuの事後分布のパラメータを計算:式(4.99)
beta_hat_k = np.sum(s_nk, axis=0) + beta
m_hat_kd = (np.dot(s_nk.T, x_nd) + beta * m_d) / beta_hat_k.reshape(K, 1)
print(beta_hat_k)
print(m_hat_kd)

# 初期値によるlambdaの事後分布のパラメータを計算:式(4.103)
w_hat_kdd = np.zeros((K, D, D))
for k in range(K):
    inv_w_dd = np.dot(s_nk[:, k] * x_nd.T, x_nd)
    inv_w_dd += beta * np.dot(m_d.reshape((D, 1)), m_d.reshape((1, D)))
    inv_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape((D, 1)), m_hat_kd[k].reshape((1, D)))
    inv_w_dd += np.linalg.inv(w_dd)
    w_hat_kdd[k] = np.linalg.inv(inv_w_dd)
nu_hat_k = np.sum(s_nk, axis=0) + nu
print(w_hat_kdd)
print(nu_hat_k)

# 初期値によるpiの事後分布のパラメータを計算:式(4.45)
alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
print(alpha_hat_k)

#%%

# 初期値によるmuの事後分布を計算
posterior_density_kg = np.empty((K, mu_point.shape[0]))
for k in range(K):
    # クラスタkのmuの事後分布を計算
    posterior_density_kg[k] = multivariate_normal.pdf(
        x=mu_point, 
        mean=m_hat_kd[k], 
        cov=np.linalg.inv(beta_hat_k[k] * nu_hat_k[k] * w_hat_kdd[k])
    )

# 初期値によるmuの事後分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], label='true val', 
            color='red', s=100, marker='x') # 真の平均
for k in range(K):
    plt.contour(mu_0_grid, mu_1_grid, posterior_density_kg[k].reshape(mu_dim)) # 事後分布
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.colorbar()
plt.legend()
plt.show()

#%%

# 初期値による混合分布を計算
init_density = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=m_hat_kd[k], cov=np.linalg.inv(nu_hat_k[k] * w_hat_kdd[k])
    )
    
    # K個の分布の加重平均を計算
    init_density += alpha_hat_k[k] / np.sum(alpha_hat_k) * tmp_density

# 初期値による分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), 
            alpha=0.5, linestyles='dashed') # 真の分布
plt.contour(x_1_grid, x_2_grid, init_density.reshape(x_dim)) # 期待値による分布:(等高線)
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.show()

#%%

# クラスタ番号を抽出
_, s_n = np.where(s_nk == 1)

# クラスタの散布図を作成
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), 
            alpha=0.5, linestyles='dashed') # 真の分布
#plt.contour(x_1_grid, x_2_grid, init_density.reshape(x_dim)) # 期待値による分布:(等高線)
plt.contourf(x_1_grid, x_2_grid, init_density.reshape(x_dim), alpha=0.5) # 期待値による分布:(塗りつぶし)
for k in range(K):
    k_idx, = np.where(s_n == k) # クラスタkのデータのインデックス
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1)) # サンプルしたクラスタ
plt.suptitle('Gaussian Mixture Model', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

#%%

## 推論処理

# 試行回数を指定
MaxIter = 150

# 途中計算に用いる項の受け皿を作成
density_st_k = np.zeros(K)

# 推移の確認用の受け皿を作成
_, s_n = np.where(s_nk == 1) # クラスタ番号を抽出
trace_s_in = [s_n.copy()]
trace_beta_ik = [beta_hat_k.copy()]
trace_m_ikd = [m_hat_kd.copy()]
trace_w_ikdd = [w_hat_kdd.copy()]
trace_nu_ik = [nu_hat_k.copy()]
trace_alpha_ik = [alpha_hat_k.copy()]

# 崩壊型ギブスサンプリング
for i in range(MaxIter):
    for n in range(N):
        
        # n番目のデータの潜在変数を初期化
        s_nk[n] = np.repeat(0, K)
        
        # muの事後分布のパラメータを計算:式(4.128)
        beta_hat_k = np.sum(s_nk, axis=0) + beta
        m_hat_kd = (np.dot(s_nk.T, x_nd) + beta * m_d) / beta_hat_k.reshape(K, 1)
        
        # lambdaの事後分布のパラメータを計算:式(4.128)
        term_m_dd = beta * np.dot(m_d.reshape((D, 1)), m_d.reshape((1, D)))
        for k in range(K):
            inv_w_dd = np.dot(s_nk[:, k] * x_nd.T, x_nd)
            inv_w_dd += term_m_dd
            inv_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape((D, 1)), m_hat_kd[k].reshape((1, D)))
            inv_w_dd += np.linalg.inv(w_dd)
            w_hat_kdd[k] = np.linalg.inv(inv_w_dd)
            nu_hat_k = np.sum(s_nk, axis=0) + nu
        
        # piの事後分布のパラメータを計算:式(4.73)
        alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
        
        # スチューデントのt分布のパラメータを計算
        nu_st_hat_k = 1 - D + nu_hat_k
        mu_st_hat_kd = m_hat_kd.copy()
        term_lmd_k = nu_st_hat_k * beta_hat_k / (1 + beta_hat_k)
        lambda_st_hat_kdd = term_lmd_k.reshape((K, 1, 1)) * w_hat_kdd
        
        # スチューデントのt分布の確率密度を計算:式(4.129)
        for k in range(K):
            density_st_k[k] = multivariate_t.pdf(
                x=x_nd[n], loc=mu_st_hat_kd[k], shape=np.linalg.inv(lambda_st_hat_kdd[k]), df=nu_st_hat_k[k]
            )
        
        # カテゴリ分布のパラメータを計算:式(4.75)
        eta_k = alpha_hat_k / np.sum(alpha_hat_k)
        
        # 潜在変数のサンプリング確率を計算:式(4.124)
        tmp_p_k = density_st_k * eta_k
        prob_s_k = tmp_p_k / np.sum(tmp_p_k) # 正規化
        
        # n番目のデータの潜在変数をサンプル
        s_nk[n] = np.random.multinomial(n=1, pvals=prob_s_k, size=1).flatten()
    
    # muの事後分布のパラメータを計算:式(4.99)
    beta_hat_k = np.sum(s_nk, axis=0) + beta
    m_hat_kd = (np.dot(s_nk.T, x_nd) + beta * m_d) / beta_hat_k.reshape(K, 1)
    
    # lambdaの事後分布のパラメータを計算:式(4.103)
    term_m_dd = beta * np.dot(m_d.reshape((D, 1)), m_d.reshape((1, D)))
    for k in range(K):
        inv_w_dd = np.dot(s_nk[:, k] * x_nd.T, x_nd)
        inv_w_dd += term_m_dd
        inv_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape((D, 1)), m_hat_kd[k].reshape((1, D)))
        inv_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(inv_w_dd)
    nu_hat_k = np.sum(s_nk, axis=0) + nu
    
    # piの事後分布のパラメータを計算:式(4.45)
    alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
    
    # i回目のパラメータを記録
    _, s_n = np.where(s_nk == 1) # クラスタ番号を抽出
    trace_s_in.append(s_n.copy())
    trace_beta_ik.append(beta_hat_k.copy())
    trace_m_ikd.append(m_hat_kd.copy())
    trace_w_ikdd.append(w_hat_kdd.copy())
    trace_nu_ik.append(nu_hat_k.copy())
    trace_alpha_ik.append(alpha_hat_k.copy())
    
    # 動作確認
    print(str(i + 1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

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

# muの事後分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], label='true val', 
            color='red', s=100, marker='x') # 真の平均
for k in range(K):
    plt.contour(mu_0_grid, mu_1_grid, posterior_density_kg[k].reshape(mu_dim)) # 事後分布
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('iter:' + str(MaxIter) + ', N=' + str(N) + ', K=' + str(K), loc='left')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.colorbar()
plt.legend()
plt.show()

#%%

# 最後に更新したパラメータの期待値による混合分布を計算
res_density = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=m_hat_kd[k], cov=np.linalg.inv(nu_hat_k[k] * w_hat_kdd[k])
    )
    
    # K個の分布の加重平均を計算
    res_density += alpha_hat_k[k] / np.sum(alpha_hat_k) * tmp_density

# 最後に更新したパラメータの期待値による分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), 
            alpha=0.5, linestyles='dashed') # 真の分布
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], 
            color='red', s=100, marker='x') # 真の平均
plt.contourf(x_1_grid, x_2_grid, res_density.reshape(x_dim), alpha=0.5) # 期待値による分布:(塗りつぶし)
for k in range(K):
    k_idx, = np.where(s_n == k) # クラスタkのデータのインデックス
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1)) # サンプルしたクラスタ
#plt.contour(x_1_grid, x_2_grid, res_density.reshape(x_dim)) # 期待値による分布:(等高線)
#plt.colorbar() # 等高線の値:(等高線用)
plt.suptitle('Gaussian Mixture Model:Collapsed Gibbs Sampling', fontsize=20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', \pi=[' + ', '.join([str(pi) for pi in np.round(alpha_hat_k / np.sum(alpha_hat_k), 3)]) + ']$', 
          loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

#%%

## 超パラメータの更新値の推移の確認

# mの推移を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    for d in range(D):
        plt.plot(np.arange(MaxIter+1), np.array(trace_m_ikd)[:, k, d], 
                 label='k=' + str(k + 1) + ', d=' + str(d + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Collapsed Gibbs Sampling', fontsize=20)
plt.title('$\hat{m}$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# betaの推移を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_beta_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Collapsed Gibbs Sampling', fontsize=20)
plt.title('$\hat{\\beta}$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# nuの推移を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_nu_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Collapsed Gibbs Sampling', fontsize=20)
plt.title('$\hat{\\nu}$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# wの推移を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    for d1 in range(D):
        for d2 in range(D):
            plt.plot(np.arange(MaxIter + 1), np.array(trace_w_ikdd)[:, k, d1, d2], 
                     alpha=0.5, label='k=' + str(k + 1) + ', d=' + str(d1 + 1) + ', d''=' + str(d2 + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Collapsed Gibbs Sampling', fontsize=20)
plt.title('$\hat{w}$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# alphaの推移を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_alpha_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Collapsed Gibbs Sampling', fontsize=20)
plt.title('$\hat{\\alpha}$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
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
    
    # i回目のmuの事後分布を作図
    plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], label='true val', 
                color='red', s=100, marker='x') # 真の平均
    for k in range(K):
        plt.contour(mu_0_grid, mu_1_grid, posterior_density_kg[k].reshape(mu_dim)) # 事後分布
    plt.suptitle('Gaussian Distribution:Collapsed Gibbs Sampling', fontsize=20)
    plt.title('iter:' + str(i) + ', N=' + str(N) + ', K=' + str(K), loc='left')
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')
    plt.legend()

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=MaxIter + 1, interval=100)
posterior_anime.save("ch4_4_4_Posterior.gif")

#%%

## サンプルによる分布の推移の確認

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_model(i):
    # i回目の混合比率を計算
    pi_hat_k = trace_alpha_ik[i] / np.sum(trace_alpha_ik[i])
    
    # i回目のサンプルによる混合分布を計算
    res_density = 0
    for k in range(K):
        # クラスタkの分布の確率密度を計算
        tmp_density = multivariate_normal.pdf(
            x=x_point, 
            mean=trace_m_ikd[i][k], 
            cov=np.linalg.inv(trace_nu_ik[i][k] * trace_w_ikdd[i][k])
        )
        
        # K個の分布の加重平均を計算
        res_density += pi_hat_k[k] * tmp_density
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のサンプルによる分布を作図
    plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), 
                alpha=0.5, linestyles='dashed') # 真の分布
    plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], 
                color='red', s=100, marker='x') # 真の平均
    #plt.contour(x_1_grid, x_2_grid, res_density.reshape(x_dim)) # 期待値による分布:(等高線)
    plt.contourf(x_1_grid, x_2_grid, res_density.reshape(x_dim), alpha=0.5) # 期待値による分布:(塗りつぶし)
    for k in range(K):
        k_idx, = np.where(trace_s_in[i] == k)
        plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1)) # サンプルしたクラスタ
        plt.suptitle('Gaussian Mixture Model:Collapsed Gibbs Sampling', fontsize=20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', \pi=[' + ', '.join([str(pi) for pi in np.round(pi_hat_k, 3)]) + ']$', 
              loc='left')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

# gif画像を作成
model_anime = animation.FuncAnimation(fig, update_model, frames=MaxIter + 1, interval=100)
model_anime.save("ch4_4_4_Model.gif")


#%%

print('end')

