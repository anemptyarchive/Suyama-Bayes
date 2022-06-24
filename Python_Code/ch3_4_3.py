# 3.4.3 多次元ガウス分布の学習と予測：平均・精度が未知の場合

#%%

# 3.4.3項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t # 多次元ガウス分布, 多次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真の平均パラメータを指定
mu_truth_d = np.array([25.0, 50.0])

# (既知の)分散共分散行列を指定
sigma2_truth_dd = np.array([[600.0, -100.0], [-100.0, 400.0]])

# (既知の精度)行列を計算
lambda_truth_dd = np.linalg.inv(sigma2_truth_dd)
print(lambda_truth_dd)


# 作図用のxのx軸の値を作成
x_0_line = np.linspace(
    mu_truth_d[0] - 3 * np.sqrt(sigma2_truth_dd[0, 0]), 
    mu_truth_d[0] + 3 * np.sqrt(sigma2_truth_dd[0, 0]), 
    num=500
)

# 作図用のxのx軸の値を作成
x_1_line = np.linspace(
    mu_truth_d[1] - 3 * np.sqrt(sigma2_truth_dd[1, 1]), 
    mu_truth_d[1] + 3 * np.sqrt(sigma2_truth_dd[1, 1]), 
    num=500
)

# 格子状のxの値を作成
x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)

# xの点を作成
x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)
x_dims = x_0_grid.shape
print(x_dims)

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_truth_dd)
)

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' + 
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# 多次元ガウス分布に従うデータを生成
x_nd = np.random.multivariate_normal(
    mean=mu_truth_d, cov=np.linalg.inv(lambda_truth_dd), size=N
)
print(x_nd[:5])

#%%

# 観測データの散布図を作成
plt.figure(figsize=(12, 9))
plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1]) # 観測データ
plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims)) # 真の分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' + 
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

## 事前分布(ガウス・ウィシャート分布)の設定

# muの事前分布のパラメータを指定
m_d = np.array([0.0, 0.0])
beta = 1.0

# lambdaの事前分布のパラメータを指定
nu = 2.0
w_dd = np.array([[0.0005, 0], [0, 0.0005]])

# lambdaの期待値を計算:式(2.89)
E_lambda_dd = nu * w_dd


# muの事前分布の分散共分散行列を計算
E_sigma2_mu_dd = np.linalg.inv(beta * E_lambda_dd)

# 作図用のmuのx軸の値を作成
mu_0_line = np.linspace(
    mu_truth_d[0] - 2 *  np.sqrt(E_sigma2_mu_dd[0, 0]), 
    mu_truth_d[0] + 2 *  np.sqrt(E_sigma2_mu_dd[0, 0]), 
    num=500
)

# 作図用のmuのy軸の値を作成
mu_1_line = np.linspace(
    mu_truth_d[1] - 2 *  np.sqrt(E_sigma2_mu_dd[1, 1]), 
    mu_truth_d[1] + 2 *  np.sqrt(E_sigma2_mu_dd[1, 1]), 
    num=500
)

# 格子状の点を作成
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)

# muの点を作成
mu_point_arr = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dims = mu_0_grid.shape
print(mu_dims)


# muの事前分布を計算:式(2.72)
prior_mu = multivariate_normal.pdf(
    x=mu_point_arr, mean=m_d, cov=np.linalg.inv(E_lambda_dd)
)

#%%

# muの事前分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
            color='red', s=100, marker='x') # 真の値
plt.contour(mu_0_grid, mu_1_grid, prior_mu.reshape(mu_dims)) # muの事前分布
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\\beta=' + str(beta) + 
          ', m=[' + ', '.join([str(m) for m in m_d]) + ']' + 
          ', E[\Lambda]=' + str([list(lmd_d) for lmd_d in np.round(E_lambda_dd, 5)]) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

# muの期待値を計算:式(2.76)
E_mu_d = m_d

# 事前分布の期待値を用いた分布を計算:式(2.72)
prior_lambda = multivariate_normal.pdf(
    x=x_point_arr, mean=E_mu_d, cov=np.linalg.inv(beta * E_lambda_dd)
)


# 事前分布の期待値を用いた分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 真の分布
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
            color='red', s=100, marker='x') # 真のmu
plt.contour(x_0_grid, x_1_grid, prior_lambda.reshape(x_dims)) # 事前分布の期待値を用いた分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$m=[' + ', '.join([str(m) for m in m_d]) + ']' +  
          ', \\nu=' + str(nu) + 
          ', W=' + str([list(w_d) for w_d in np.round(w_dd, 5)]) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

## 事後分布(ガウス・ウィシャート分布)の計算

# muの事後分布のパラメータを計算:式(3.129)
beta_hat = N + beta
m_hat_d = (np.sum(x_nd, axis=0) + beta * m_d) / beta_hat

# lambdaの事後分布のパラメータを計算:式(3.133)
term_x_dd = np.dot(x_nd.T, x_nd)
term_m_dd = beta * np.dot(m_d.reshape([2, 1]), m_d.reshape([1, 2]))
term_m_hat_dd = beta_hat * np.dot(m_hat_d.reshape([2, 1]), m_hat_d.reshape([1, 2]))
w_hat_dd = np.linalg.inv(
  term_x_dd + term_m_dd - term_m_hat_dd + np.linalg.inv(w_dd)
)
nu_hat = N + nu


# lambdaの期待値を計算:式(2.89)
E_lambda_hat_dd = nu_hat * w_hat_dd

# muの事後分布を計算:式(2.72)
posterior_mu = multivariate_normal.pdf(
    x=mu_point_arr, mean=m_hat_d, cov=np.linalg.inv(beta_hat * E_lambda_hat_dd)
)

#%%

# muの事後分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
            color='red', s=100, marker='x') # 真の値
plt.contour(mu_0_grid, mu_1_grid, posterior_mu.reshape(mu_dims)) # muの事後分布
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' + 
          ', \hat{\\beta}=' + str(beta_hat) + 
          ', E[\hat{\Lambda}]=' + str([list(lmd_d) for lmd_d in np.round(E_lambda_hat_dd, 5)]) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

# muの期待値を計算:式(2.76)
E_mu_hat_d = m_hat_d

# 事後分布の期待値を用いた分布を計算:式(2.72)
posterior_lambda = multivariate_normal.pdf(
    x=x_point_arr, mean=E_mu_hat_d, cov=np.linalg.inv(E_lambda_hat_dd)
)


# 事後分布の期待値を用いた分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 真の分布
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
            color='red', s=100, marker='x') # 真のmu
plt.contour(x_0_grid, x_1_grid, posterior_lambda.reshape(x_dims)) # 事後分布の期待を用いた分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' + 
          ', \hat{\\nu}=' + str(nu_hat) + 
          ', \hat{W}=' + str([list(w_d) for w_d in np.round(w_hat_dd, 5)]) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

## 予測分布(多次元スチューデントのt分布)の計算

# 次元数を取得
D = len(m_d)

# 予測分布のパラメータを計算:式(3.140')
mu_s_d = m_hat_d
lambda_s_hat_dd = (1.0 - D + nu_hat) * beta_hat / (1 + beta_hat) * w_hat_dd
nu_s_hat = 1.0 - D + nu_hat


# 予測分布を計算:式(3.121)
predict = multivariate_t.pdf(
    x=x_point_arr, loc=mu_s_d, shape=np.linalg.inv(lambda_s_hat_dd), df=nu_s_hat
)

#%%

# 予測分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 真の分布
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1]) # 観測データ
plt.contour(x_0_grid, x_1_grid, predict.reshape(x_dims)) # 予測分布
plt.ylabel('$x_2$')
plt.suptitle("Multivariate Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{\mu}_s=[' + ', '.join([str(mu) for mu in np.round(mu_s_d, 1)]) + ']' + 
          ', \hat{\Lambda}_s=' + str([list(lmd_d) for lmd_d in np.round(lambda_s_hat_dd, 5)]) + 
          ', \hat{\\nu}_s=' + str(nu_s_hat) + '$', 
          loc='left')
plt.colorbar()
plt.show()

#%%

### ・アニメーションによる推移の確認

# 3.4.3項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t # 多次元ガウス分布, 多次元スチューデントのt分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# 真の平均パラメータを指定
mu_truth_d = np.array([25.0, 50.0])

# (既知の)分散共分散行列を指定
sigma2_truth_dd = np.array([[600.0, 100.0], [100.0, 400.0]])

# (既知の精度)行列を計算
lambda_truth_dd = np.linalg.inv(sigma2_truth_dd)


# muの事前分布のパラメータを指定
m_d = np.array([0.0, 0.0])
beta = 1

# lambdaの事前分布のパラメータを指定
nu = 2
w_dd = np.array([[0.0005, 0], [0, 0.0005]])
inv_w_dd = np.linalg.inv(w_dd)


# 初期値による予測分布のパラメータを計算:式(3.140)
mu_s_d = m_d
lambda_s_dd = (nu - 1.0) * beta / (1 + beta) * w_dd
nu_s = nu - 1.0


# 作図用のmuのx軸の値を作成
mu_0_line = np.linspace(
    mu_truth_d[0] - 2 *  np.sqrt(np.linalg.inv(beta * nu * w_dd)[0, 0]), 
    mu_truth_d[0] + 2 *  np.sqrt(np.linalg.inv(beta * nu * w_dd)[0, 0]), 
    num=500
)

# 作図用のmuのy軸の値を作成
mu_1_line = np.linspace(
    mu_truth_d[1] - 2 *  np.sqrt(np.linalg.inv(beta * nu * w_dd)[1, 1]), 
    mu_truth_d[1] + 2 *  np.sqrt(np.linalg.inv(beta * nu * w_dd)[1, 1]), 
    num=500
)

# 格子状の点を作成
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)

# muの点を作成
mu_point_arr = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dims = mu_0_grid.shape


# 作図用のxのx軸の値を作成
x_0_line = np.linspace(
    mu_truth_d[0] - 3 * np.sqrt(sigma2_truth_dd[0, 0]), 
    mu_truth_d[0] + 3 * np.sqrt(sigma2_truth_dd[0, 0]), 
    num=500
)

# 作図用のxのx軸の値を作成
x_1_line = np.linspace(
    mu_truth_d[1] - 3 * np.sqrt(sigma2_truth_dd[1, 1]), 
    mu_truth_d[1] + 3 * np.sqrt(sigma2_truth_dd[1, 1]), 
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
trace_beta = [beta]
trace_posterior = [
    multivariate_normal.pdf(
        x=mu_point_arr, mean=m_d, cov=np.linalg.inv(beta * nu * w_dd)
    )
]
trace_mu_s = [mu_s_d]
trace_lambda_s = [lambda_s_dd]
trace_nu_s = [nu_s]
trace_predict = [
    multivariate_t.pdf(
        x=x_point_arr, loc=mu_s_d, shape=np.linalg.inv(lambda_s_dd), df=nu_s
    )
]

# ベイズ推論
for n in range(N):
    # 多次元ガウス分布に従うデータを生成
    x_nd[n] = np.random.multivariate_normal(
        mean=mu_truth_d, cov= np.linalg.inv(lambda_truth_dd)
    ).flatten()
    
    # muの事後分布のパラメータを:式(3.129)
    old_beta = beta
    old_m_d = m_d.copy()
    beta += 1
    m_d = (x_nd[n] + old_beta * m_d) / beta
    
    # lambdaの事後分布のパラメータを更新:式(1.33)
    term_x_dd = np.dot(x_nd[n].reshape([2, 1]), x_nd[n].reshape([1, 2]))
    old_term_m_dd = old_beta * np.dot(old_m_d.reshape([2, 1]), old_m_d.reshape([1, 2]))
    term_m_dd = beta * np.dot(m_d.reshape([2, 1]), m_d.reshape([1, 2]))
    inv_w_dd += term_x_dd + old_term_m_dd - term_m_dd
    nu += 1
    
    # muの事後分布を計算:式(2.72)
    trace_posterior.append(
        multivariate_normal.pdf(
            x=mu_point_arr, mean=m_d, cov=np.linalg.inv(beta * nu * np.linalg.inv(inv_w_dd))
        )
    )
    
    # 予測分布のパラメータを更新:式(3.140)
    mu_s_d = m_d
    lambda_s_dd = (nu - 1.0) * beta / (1 + beta) * np.linalg.inv(inv_w_dd)
    nu_s = nu - 1.0
    
    # 予測分布を計算:式(3.121)
    trace_predict.append(
        multivariate_t.pdf(
            x=x_point_arr, loc=mu_s_d, shape=np.linalg.inv(lambda_s_dd), df=nu_s
        )
    )
    
    # n回目の結果を記録
    trace_m.append(m_d)
    trace_beta.append(beta)
    trace_mu_s.append(mu_s_d)
    trace_lambda_s.append(lambda_s_dd)
    trace_nu_s.append(nu_s)
    
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
    plt.contour(mu_0_grid, mu_1_grid, trace_posterior[n].reshape(mu_dims)) # muの事後分布
    plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], 
                color='red', s=100, marker='x') # 真の値
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')
    plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
    plt.title('$n=' + str(n) + 
              ', \hat{\\beta}=' + str(trace_beta[n]) + 
              ', \hat{m}=[' + ', '.join([str(m) for m in np.round(trace_m[n], 1)]) + ']' + '$', 
              loc='left')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_4_3_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_truth_dd)
)

# 画像サイズを指定
fig = plt.figure(figsize=(7, 7))

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
    plt.contour(x_0_grid, x_1_grid, trace_predict[n].reshape(x_dims)) # 予測分布
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.suptitle("Multivariate Student's t Distribution", fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\mu}_s=[' + ', '.join([str(mu) for mu in np.round(trace_mu_s[n], 1)]) + ']' + 
              ', \hat{\Lambda}_s=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_s[n], 5)]) + 
              ', \hat{\\nu}_s=' + str(trace_nu_s[n]) + '$', 
              loc='left')

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_4_3_Predict.gif")

#%%

print('end')

