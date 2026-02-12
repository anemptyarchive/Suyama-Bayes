
# 多次元ガウスモデル -------------------------------------------------------------

# chapter 3.4.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#%%

## モデルの設定

# 真の平均パラメータを指定
mu_truth_d = np.array([25.0, 50.0])

# (既知の)分散共分散行列を指定
sigma2_dd = np.array([[600.0, 100.0], [100.0, 400.0]])

# (既知の精度)行列を計算
lambda_dd = np.linalg.inv(sigma2_dd)


# muの事前分布の平均パラメータを指定
m_d = np.array([0.0, 0.0])

# muの事前分布の分散共分散行列を指定
sigma2_mu_dd = np.array([[1000.0, 0.0], [0.0, 1000.0]])

# muの事前分布の精度行列を計算
lambda_mu_dd = np.linalg.inv(sigma2_mu_dd)


# 初期値による予測分布のパラメータを計算:式(3.109),(3.110)
lambda_star_dd = np.linalg.inv(
    np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_dd)
)
mu_star_d = m_d


# 作図用のmuのx軸の値を作成
mu_0_line = np.linspace(
    mu_truth_d[0] - 2 *  np.sqrt(sigma2_mu_dd[0, 0]), 
    mu_truth_d[0] + 2 *  np.sqrt(sigma2_mu_dd[0, 0]), 
    num=500
)

# 作図用のmuのy軸の値を作成
mu_1_line = np.linspace(
    mu_truth_d[1] - 2 *  np.sqrt(sigma2_mu_dd[1, 1]), 
    mu_truth_d[1] + 2 *  np.sqrt(sigma2_mu_dd[1, 1]), 
    num=500
)

# 格子状の点を作成
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)

# muの点を作成
mu_point_arr = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dims = mu_0_grid.shape


# 作図用のxのx軸の値を作成
x_0_line = np.linspace(
    mu_truth_d[0] - 3 * np.sqrt(sigma2_dd[0, 0]), 
    mu_truth_d[0] + 3 * np.sqrt(sigma2_dd[0, 0]), 
    num=500
)

# 作図用のxのx軸の値を作成
x_1_line = np.linspace(
    mu_truth_d[1] - 3 * np.sqrt(sigma2_dd[1, 1]), 
    mu_truth_d[1] + 3 * np.sqrt(sigma2_dd[1, 1]), 
    num=500
)

# 格子状のxの値を作成
x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)

# xの点を作成
x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)
x_dims = x_0_grid.shape

#%%

## 推論処理

# データ数(試行回数)を指定
N = 100

# 観測データの受け皿を作成
x_nd = np.empty((N, 2))

# 推移の記録用の受け皿を初期化
trace_m = [m_d]
trace_lambda_mu = [lambda_mu_dd]
trace_posterior = [
    multivariate_normal.pdf(
        x=mu_point_arr, mean=m_d, cov=np.linalg.inv(lambda_mu_dd)
    )
]
trace_mu_star = [mu_star_d]
trace_lambda_star = [lambda_star_dd]
trace_predict = [
    multivariate_normal.pdf(
        x=x_point_arr, mean=mu_star_d, cov=np.linalg.inv(lambda_star_dd)
    )
]

# ベイズ推論
for n in range(N):
    # 多次元ガウス分布に従うデータを生成
    x_nd[n] = np.random.multivariate_normal(
        mean=mu_truth_d, cov=np.linalg.inv(lambda_dd), size=1
    ).flatten()
    
    # muの事後分布のパラメータを更新:式(3.102),(3.102)
    old_lambda_mu_dd = lambda_mu_dd.copy()
    lambda_mu_dd += lambda_dd
    term_m_d = np.dot(lambda_dd, x_nd[n]) + np.dot(old_lambda_mu_dd, m_d)
    m_d = np.dot(np.linalg.inv(lambda_mu_dd), term_m_d)
    
    # muの事後分布(多次元ガウス分布)を計算:式(2.72)
    trace_posterior.append(
        multivariate_normal.pdf(
            x=mu_point_arr, mean=m_d, cov=np.linalg.inv(lambda_mu_dd)
        )
    )
    
    # 予測分布のパラメータを計算:式(3.109),(3.110)
    lambda_star_dd = np.linalg.inv(
        np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_dd)
    )
    mu_star_d = m_d
    
    # 予測分布を計算:式(2.72)
    trace_predict.append(
        multivariate_normal.pdf(
            x=x_point_arr, mean=mu_star_d, cov=np.linalg.inv(lambda_star_dd)
        )
    )
    
    # n回目の結果を記録
    trace_m.append(m_d)
    trace_lambda_mu.append(lambda_mu_dd)
    trace_mu_star.append(mu_star_d)
    trace_lambda_star.append(lambda_star_dd)
    
    # 動作確認
    print('n=' + str(n + 1) + ' (' + str(np.round((n + 1) / N * 100, 1)) + '%)')

#%%

## muの事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目のmuの事後分布を作図
    plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
                color='red', s=100, marker='x') # 真の値
    plt.contour(mu_0_grid, mu_1_grid, trace_posterior[n].reshape(mu_dims)) # muの事後分布
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')
    plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{m}=[' + ', '.join([str(m) for m in np.round(trace_m[n], 1)]) + ']' + 
              ', \hat{\Lambda}_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_mu[n], 5)]) + '$', 
              loc='left')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_4_1_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_dd)
)

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims), 
                alpha=0.5, linestyles='--') # 真の分布
    plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
                color='red', s=100, marker='x') # 真のmu
    plt.scatter(x=x_nd[:n, 0], y=x_nd[:n, 1]) # 観測データ
    plt.contour(x_0_grid, x_１_grid, trace_predict[n].reshape(x_dims)) # 予測分布
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\mu}_{*}=[' + ', '.join([str(mu) for mu in np.round(trace_mu_star[n], 1)]) + ']' + 
              ', \hat{\Lambda}_{*}=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_star[n], 5)]) + '$', 
              loc='left')

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_4_1_Predict.gif")

#%%

print('end')
