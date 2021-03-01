# 3.4.2 多次元ガウス分布の学習と予測：精度が未知の場合

#%%

# 3.4.2項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t # 多次元ガウス分布, 多次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真のパラメータを指定
mu_d = np.array([25.0, 50.0])
sigma_truth_dd = np.array([[15.0, 10.0], [10.0, 20.0]])
lambda_truth_dd = np.linalg.inv(sigma_truth_dd**2)
print(lambda_truth_dd)

# 作図用のxの点を作成
x_1_point = np.linspace(mu_d[0] - 4 * sigma_truth_dd[0, 0], mu_d[0] + 4 * sigma_truth_dd[0, 0], num=300)
x_2_point = np.linspace(mu_d[1] - 4 * sigma_truth_dd[1, 1], mu_d[1] + 4 * sigma_truth_dd[1, 1], num=300)
x_1_grid, x_2_grid = np.meshgrid(x_1_point, x_2_point)
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape
print(x_dims)

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_d, cov=np.linalg.inv(lambda_truth_dd)
)
print(true_model)

#%%

# 尤度を作図
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\mu=[' + ', '.join([str(mu) for mu in mu_d]) + ']' + 
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 100

# 多次元ガウス分布に従うデータを生成
x_nd = np.random.multivariate_normal(
    mean=mu_d, cov=np.linalg.inv(lambda_truth_dd), size=N
)
print(x_nd[:5])

#%%

# 観測データの散布図を作成
plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1]) # 観測データ
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_d]) + ']' + 
          ', \Sigma_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(np.sqrt(np.linalg.inv(lambda_truth_dd)), 1)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 事前分布(ウィシャート分布)の設定

# lambdaの事前分布のパラメータを指定
w_dd = np.array([[0.00005, 0], [0, 0.00005]])
nu = 2

# lambdaの期待値を計算:式(2.89)
E_lambda_dd = nu * w_dd

# 事前分布の期待値を用いた分布を計算:式(2.72)
prior = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_d, cov=np.linalg.inv(E_lambda_dd)
)
print(prior)

#%%

# 事前分布の期待値を用いた分布を作図
plt.contour(x_1_grid, x_2_grid, prior.reshape(x_dims)) # lambdaの期待値による分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\\nu=' + str(nu) + 
          ', W=' + str([list(w_d) for w_d in np.round(w_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 事後分布(ウィシャート分布)の設定

# lambdaの事後分布のパラメータを計算:式(3.116)
w_hat_dd = np.linalg.inv(
  np.dot((x_nd - mu_d).T, (x_nd - mu_d)) + np.linalg.inv(w_dd)
)
nu_hat = N + nu
print(w_hat_dd)
print(nu_hat)

# lambdaの期待値を計算:式(2.89)
E_lambda_hat_dd = nu_hat * w_hat_dd
print(E_lambda_hat_dd)

# 事後分布の期待値を用いた分布を計算:式(2.72)
posterior = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_d, cov=np.linalg.inv(E_lambda_hat_dd)
)
print(posterior)

#%%

# 事後分布の期待値を用いた分布を作図
plt.contour(x_1_grid, x_2_grid, posterior.reshape(x_dims)) # lambdaの期待値による分布
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\\nu}=' + str(nu_hat) + 
          ', \hat{W}=' + str([list(w_d) for w_d in np.round(w_hat_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 予測分布(多次元スチューデントのt分布)の計算

# 次元数:(固定)
D = 2.0

# 予測分布のパラメータを計算:式(3.124')
mu_s_d = mu_d
lambda_s_hat_dd = (1.0 - D + nu_hat) * w_hat_dd
nu_s_hat = 1.0 - D + nu_hat

# 予測分布を計算:式(3.121)
predict = multivariate_t.pdf(
    x=x_point_arr, loc=mu_s_d, shape=np.linalg.inv(lambda_s_hat_dd), df=nu_s_hat
)
print(predict)

#%%

# 予測分布を作図
plt.contour(x_1_grid, x_2_grid, predict.reshape(x_dims)) # 予測分布
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle("Multivariate Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\\nu}_s=' + str(nu_s_hat) + 
          ', \mu_s=[' + ', '.join([str(mu) for mu in mu_s_d]) + ']' + 
          ', \hat{\Lambda}_s=' + str([list(lmd_d) for lmd_d in np.round(lambda_s_hat_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

### ・アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t # 多次元ガウス分布, 多次元スチューデントのt分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## 尤度(ガウス分布)の設定

# 真のパラメータを指定
mu_d = np.array([25.0, 50.0])
sigma_truth_dd = np.array([[15.0, 10.0], [10.0, 20.0]])
lambda_truth_dd = np.linalg.inv(sigma_truth_dd**2)

# lambdaの事前分布のパラメータを指定
w_dd = np.array([[0.00005, 0], [0, 0.00005]])
inv_w_dd = np.linalg.inv(w_dd)
nu = 2.0

# lambdaの期待値を計算:式(2.89)
E_lambda_dd = nu * w_dd

# 初期値による予測分布のパラメータを計算:式(3.124)
mu_s_d = mu_d
lambda_s_dd = (nu - 1.0) * w_dd
nu_s = nu - 1.0

# データ数(試行回数)を指定
N = 150

# 作図用のxの点を作成
x_1_point = np.linspace(mu_d[0] - 4 * sigma_truth_dd[0, 0], mu_d[0] + 4 * sigma_truth_dd[0, 0], num=300)
x_2_point = np.linspace(mu_d[1] - 4 * sigma_truth_dd[1, 1], mu_d[1] + 4 * sigma_truth_dd[1, 1], num=300)
x_1_grid, x_2_grid = np.meshgrid(x_1_point, x_2_point)
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape

# 推移の記録用の受け皿を初期化
x_nd = np.empty((N, 2))
trace_w = [[list(w_d) for w_d in w_dd]]
trace_nu = [nu]
trace_posterior = [
    multivariate_normal.pdf(
        x=x_point_arr, mean=mu_d, cov=np.linalg.inv(E_lambda_dd)
    )
]
trace_lambda_s = [[list(lmd_d) for lmd_d in lambda_s_dd]]
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
        mean=mu_d, cov=np.linalg.inv(lambda_truth_dd), size=1
    ).flatten()
    
    # lambdaの事後分布のパラメータを更新:式(3.116)
    inv_w_dd += np.dot((x_nd[n] - mu_d).reshape([2, 1]), (x_nd[n] - mu_d).reshape([1, 2]))
    nu += 1
    
    # lambdaの期待値を計算:式(2.89)
    E_lambda_dd = nu * np.linalg.inv(inv_w_dd)
    
    # 事後分布の期待値を用いた分布を計算:式(2.72)
    trace_posterior.append(
        multivariate_normal.pdf(
            x=x_point_arr, mean=mu_d, cov=np.linalg.inv(E_lambda_dd)
        )
    )
    
    # 予測分布のパラメータを更新:式(3.124)
    #mu_s_d = mu_d
    lambda_s_dd = (nu - 1.0) * np.linalg.inv(inv_w_dd)
    nu_s = nu - 1.0
    
    # 予測分布を計算:式(3.121)
    trace_predict.append(
        multivariate_t.pdf(
            x=x_point_arr, loc=mu_s_d, shape=np.linalg.inv(lambda_s_dd), df=nu_s
        )
    )
    
    # 超パラメータを記録
    trace_w.append([list(w_d) for w_d in np.linalg.inv(inv_w_dd)])
    trace_nu.append(nu)
    trace_lambda_s.append([list(lmd_d) for lmd_d in lambda_s_dd])
    trace_nu_s.append(nu_s)

# 観測データを確認
print(x_nd[:5])

#%%

## 作図処理

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_d, cov=np.linalg.inv(lambda_truth_dd)
)

#%%

## muの事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior_mu(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の事前分布の期待値を用いた分布を作図
    plt.contour(x_1_grid, x_2_grid, np.array(trace_posterior[n]).reshape(x_dims)) # lambdaの期待値による分布
    plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
                alpha=0.5, linestyles='--') # 尤度
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\\nu}=' + str(trace_nu[n]) + 
              ', \hat{W}=' + str([list(w_d) for w_d in np.round(trace_w[n], 5)]) + 
              '$', loc='left')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior_mu, frames=N + 1, interval=100)
posterior_anime.save("ch3_4_2_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.contour(x_1_grid, x_2_grid, np.array(trace_predict[n]).reshape(x_dims)) # 予測分布
    plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
                alpha=0.5, linestyles='--') # 尤度
    plt.scatter(x=x_nd[:n, 0], y=x_nd[:n, 1]) # 観測データ
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.suptitle("Multivariate Student's t Distribution", fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\\nu}_s=' + str(trace_nu_s[n]) + 
              ', \mu_s=[' + ', '.join([str(mu) for mu in mu_s_d]) + ']' + 
              ', \hat{\Lambda}_s=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_s[n], 5)]) + 
              '$', loc='left')

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_4_2_Predict.gif")

#%%

print('end')

