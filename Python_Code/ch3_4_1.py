# 3.4.1 多次元ガウス分布の学習と予測：平均が未知の場合

#%%

# 3.4.1項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal # 多次元ガウス分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真のパラメータを指定
mu_truth_d = np.array([25.0, 50.0])
sigma_dd = np.array([[15.0, 10.0], [10.0, 20.0]])
lambda_dd = np.linalg.inv(sigma_dd**2)
print(lambda_dd)

# 作図用のxの点を作成
x_1_point = np.linspace(mu_truth_d[0] - 4 * sigma_dd[0, 0], mu_truth_d[0] + 4 * sigma_dd[0, 0], num=300)
x_2_point = np.linspace(mu_truth_d[1] - 4 * sigma_dd[1, 1], mu_truth_d[1] + 4 * sigma_dd[1, 1], num=300)
x_1_grid, x_2_grid = np.meshgrid(x_1_point, x_2_point)
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape
print(x_dims)

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_dd)
)
print(true_model)

#%%

# 尤度を作図
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' + 
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_dd, 5)]) + '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 100

# 多次元ガウス分布に従うデータを生成
x_nd = np.random.multivariate_normal(
    mean=mu_truth_d, cov=np.linalg.inv(lambda_dd), size=N
)
print(x_nd[:5])

#%%

# 観測データの散布図を作成
plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1]) # 観測データ
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' + 
          ', \Sigma_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(np.sqrt(np.linalg.inv(lambda_dd)), 1)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 事前分布(多次元ガウス分布)の設定

# muの事前分布のパラメータを指定
m_d = np.array([0.0, 0.0])
sigma_mu_dd = np.array([[100.0, 0.0], [0.0, 100.0]])
lambda_mu_dd = np.linalg.inv(sigma_mu_dd**2)
print(lambda_mu_dd)

# 作図用のmuの点を作成
mu_1_point = np.linspace(mu_truth_d[0] - 100.0, mu_truth_d[0] + 100.0, num=300)
mu_2_point = np.linspace(mu_truth_d[1] - 100.0, mu_truth_d[1] + 100.0, num=300)
mu_1_grid, mu_2_grid = np.meshgrid(mu_1_point, mu_2_point)
mu_point_arr = np.stack([mu_1_grid.flatten(), mu_2_grid.flatten()], axis=1)
mu_dims = mu_1_grid.shape
print(mu_dims)

# muの事前分布を計算:式(2.72)
prior = multivariate_normal.pdf(
    x=mu_point_arr, mean=m_d, cov=np.linalg.inv(lambda_mu_dd)
)
print(prior)

#%%

# muの事前分布を作図
plt.contour(mu_1_grid, mu_2_grid, prior.reshape(mu_dims)) # muの事前分布
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$m=[' + ', '.join([str(m) for m in m_d]) + ']' + 
          ', \Lambda_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(lambda_mu_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 事後分布(多次元ガウス分布)の計算

# muの事後分布のパラメータを計算:式(3.102),(3.103)
lambda_mu_hat_dd = N * lambda_dd + lambda_mu_dd
term_x_d = np.dot(lambda_dd, np.sum(x_nd, axis=0))
term_m_d = np.dot(lambda_mu_dd, m_d)
m_hat_d = np.dot(np.linalg.inv(lambda_mu_hat_dd), (term_x_d + term_m_d))
print(lambda_mu_dd)
print(m_hat_d)

# muの事後分布を計算:式(2.72)
posterior = multivariate_normal.pdf(
    x=mu_point_arr, mean=m_hat_d, cov=np.linalg.inv(lambda_mu_hat_dd)
)
print(posterior)

#%%

# muの事後分布を作図
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
plt.contour(mu_1_grid, mu_2_grid, posterior.reshape(mu_dims))
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' + 
          ', \hat{\Lambda}_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(lambda_mu_hat_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 予測分布(多次元ガウス分布)の計算

# 予測分布のパラメータを計算:式(3.109'),(3.110')
lambda_star_hat_dd = np.linalg.inv(
    np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_hat_dd)
)
mu_star_hat_d = m_hat_d
print(lambda_star_hat_dd)
print(mu_star_hat_d)

# 予測分布を計算:式(2.72)
predict = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_star_hat_d, cov=np.linalg.inv(lambda_star_hat_dd)
)
print(predict)

#%%

# 予測分布を作図
plt.contour(x_1_grid, x_2_grid, predict.reshape(x_dims)) # 予測分布
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 真の分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\mu}_{*}=[' + ', '.join([str(mu) for mu in np.round(mu_star_hat_d, 1)]) + ']' + 
          ', \hat{\Lambda}_{*}=' + str([list(lmd_d) for lmd_d in np.round(lambda_star_hat_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()


#%%

### ・アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal # 多次元ガウス分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## 推論処理

# 真のパラメータを指定
mu_truth_d = np.array([25.0, 50.0])
sigma_dd = np.array([[15.0, 10.0], [10.0, 20.0]])
lambda_dd = np.linalg.inv(sigma_dd**2)

# muの事前分布のパラメータを指定
m_d = np.array([0.0, 0.0])
sigma_mu_dd = np.array([[100.0, 0.0], [0.0, 100.0]])
lambda_mu_dd = np.linalg.inv(sigma_mu_dd**2)

# 初期値による予測分布のパラメータを計算:式(3.109),(3.110)
lambda_star_dd = np.linalg.inv(
    np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_dd)
)
mu_star_d = m_d

# データ数(試行回数)を指定
N = 100

# 作図用のmuの点を作成
mu_1_point = np.linspace(mu_truth_d[0] - 100.0, mu_truth_d[0] + 100.0, num=300)
mu_2_point = np.linspace(mu_truth_d[1] - 100.0, mu_truth_d[1] + 100.0, num=300)
mu_1_grid, mu_2_grid = np.meshgrid(mu_1_point, mu_2_point)
mu_point_arr = np.stack([mu_1_grid.flatten(), mu_2_grid.flatten()], axis=1)
mu_dims = mu_1_grid.shape

# 作図用のxの点を作成
x_1_point = np.linspace(mu_truth_d[0] - 4 * sigma_dd[0, 0], mu_truth_d[0] + 4 * sigma_dd[0, 0], num=300)
x_2_point = np.linspace(mu_truth_d[1] - 4 * sigma_dd[1, 1], mu_truth_d[1] + 4 * sigma_dd[1, 1], num=300)
x_1_grid, x_2_grid = np.meshgrid(x_1_point, x_2_point)
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape

# 推移の記録用の受け皿を初期化
x_nd = np.empty((N, 2))
trace_m = [list(m_d)]
trace_lambda_mu = [[list(lmd_d) for lmd_d in lambda_mu_dd]]
trace_posterior = [
    multivariate_normal.pdf(
        x=mu_point_arr, mean=m_d, cov=np.linalg.inv(lambda_mu_dd)
    )
]
trace_mu_star = [list(mu_star_d)]
trace_lambda_star = [[list(lmd_d) for lmd_d in lambda_star_dd]]
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
    
    # 超パラメータを記録
    trace_m.append(list(m_d))
    trace_lambda_mu.append([list(lmd_d) for lmd_d in lambda_mu_dd])
    trace_mu_star.append(list(mu_star_d))
    trace_lambda_star.append([list(lmd_d) for lmd_d in lambda_star_dd])
    
    # 動作確認
    print('n=' + str(n + 1) + ' (' + str(np.round((n + 1) / N * 100, 1)) + '%)')

# 観測データを確認
print(x_nd[:5])

#%%

## muの事後分布の推移をgif画像化

## 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior_mu(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目のmuの事後分布を作図
    plt.contour(mu_1_grid, mu_2_grid, np.array(trace_posterior[n]).reshape(mu_dims)) # muの事後分布
    plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')
    plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
    plt.title('$n=' + str(n) + 
              ', \hat{m}=[' + ', '.join([str(m) for m in np.round(trace_m[n], 1)]) + ']' + 
              ', \hat{\Lambda}_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_mu[n], 5)]) + 
              '$', loc='left')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior_mu, frames=N + 1, interval=100)
posterior_anime.save("ch3_4_1_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_dd)
)

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.contour(x_1_grid, x_2_grid, np.array(trace_predict[n]).reshape(x_dims)) # 予測分布
    plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
                alpha=0.5, linestyles='--') # 真の分布
    plt.scatter(x=x_nd[:n, 0], y=x_nd[:n, 1]) # 観測データ
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\mu}_{*}=[' + ', '.join([str(mu) for mu in np.round(trace_mu_star[n], 1)]) + ']' + 
              ', \hat{\Lambda}_{*}=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_star[n], 5)]) + 
              '$', loc='left')

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_4_1_Predict.gif")

#%%

print('end')
