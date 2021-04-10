# 3.4.3 多次元ガウス分布の学習と予測：平均・精度が未知の場合

#%%

# 3.4.3項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t # 多次元ガウス分布, 多次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真のパラメータを指定
mu_truth_d = np.array([25.0, 50.0])
sigma_truth_dd = np.array([[20.0, 15.0], [15.0, 30.0]])
lambda_truth_dd = np.linalg.inv(sigma_truth_dd**2)
print(lambda_truth_dd)

# 作図用のxの点を作成
x_1_point = np.linspace(mu_truth_d[0] - 4 * sigma_truth_dd[0, 0], mu_truth_d[0] + 4 * sigma_truth_dd[0, 0], num=1000)
x_2_point = np.linspace(mu_truth_d[1] - 4 * sigma_truth_dd[1, 1], mu_truth_d[1] + 4 * sigma_truth_dd[1, 1], num=1000)
x_1_grid, x_2_grid = np.meshgrid(x_1_point, x_2_point)
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape
print(x_dims)

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_truth_dd)
)
print(true_model)

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' + 
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$', loc='left')
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
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims)) # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' + 
          ', \Sigma=' + str([list(lmd_d) for lmd_d in np.round(np.sqrt(np.linalg.inv(lambda_truth_dd)), 1)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 事前分布(ガウス・ウィシャート分布)の設定

# muの事前分布のパラメータを指定
beta = 1
m_d = np.array([0.0, 0.0])

# lambdaの事前分布のパラメータを指定
w_dd = np.array([[0.0005, 0], [0, 0.0005]])
nu = 2

# lambdaの期待値を計算:式(2.89)
E_lambda_dd = nu * w_dd
print(E_lambda_dd)

# 作図用のmuの点を作成
mu_1_point = np.linspace(mu_truth_d[0] - 100.0, mu_truth_d[0] + 100.0, num=1000)
mu_2_point = np.linspace(mu_truth_d[1] - 100.0, mu_truth_d[1] + 100.0, num=1000)
mu_1_grid, mu_2_grid = np.meshgrid(mu_1_point, mu_2_point)
mu_point_arr = np.stack([mu_1_grid.flatten(), mu_2_grid.flatten()], axis=1)
mu_dims = mu_1_grid.shape
print(mu_dims)

# muの事前分布を計算:式(2.72)
prior_mu = multivariate_normal.pdf(
    x=mu_point_arr, mean=m_d, cov=np.linalg.inv(E_lambda_dd)
)
print(prior_mu)

#%%

# muの事前分布を作図
plt.figure(figsize=(12, 9))
plt.contour(mu_1_grid, mu_2_grid, prior_mu.reshape(mu_dims)) # muの事前分布
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\\beta=' + str(beta) + ', m=[' + ', '.join([str(m) for m in m_d]) + ']' + 
          ', E[\Lambda]=' + str([list(lmd_d) for lmd_d in np.round(E_lambda_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

# muの期待値を計算:式(2.76)
E_mu_d = m_d

# 事前分布の期待値を用いた分布を計算:式(2.72)
prior_lambda = multivariate_normal.pdf(
    x=x_point_arr, mean=E_mu_d, cov=np.linalg.inv(beta * E_lambda_dd)
)
print(prior_lambda)

#%%

# 事前分布の期待値を用いた分布を作図
plt.figure(figsize=(9, 9))
plt.contour(x_1_grid, x_2_grid, prior_lambda.reshape(x_dims)) # 事前分布の期待値を用いた分布
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 尤度
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$\\nu=' + str(nu) + 
          ', W=' + str([list(w_d) for w_d in np.round(w_dd, 5)]) + 
          '$', loc='left')
plt.colorbar()
plt.show()

#%%

## 事後分布(ガウス・ウィシャート分布)の計算

# muの事後分布のパラメータを計算:式(3.129)
beta_hat = N + beta
m_hat_d = (np.sum(x_nd, axis=0) + beta * m_d) / beta_hat
print(beta_hat)
print(m_hat_d)

# lambdaの事後分布のパラメータを計算:式(3.133)
term_x_dd = np.dot(x_nd.T, x_nd)
term_m_dd = beta * np.dot(m_d.reshape([2, 1]), m_d.reshape([1, 2]))
term_m_hat_dd = beta_hat * np.dot(m_hat_d.reshape([2, 1]), m_hat_d.reshape([1, 2]))
w_hat_dd = np.linalg.inv(
  term_x_dd + term_m_dd - term_m_hat_dd + np.linalg.inv(w_dd)
)
nu_hat = N + nu
print(w_hat_dd)
print(nu_hat)

# lambdaの期待値を計算:式(2.89)
E_lambda_hat_dd = nu_hat * w_hat_dd
print(E_lambda_hat_dd)


# muの事後分布を計算:式(2.72)
posterior_mu = multivariate_normal.pdf(
    x=mu_point_arr, mean=m_hat_d, cov=np.linalg.inv(beta_hat * E_lambda_hat_dd)
)
print(posterior_mu)

#%%

# muの事後分布を作図
plt.figure(figsize=(12, 9))
plt.contour(mu_1_grid, mu_2_grid, posterior_mu.reshape(mu_dims))
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.suptitle('Multivariate Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\\beta}=' + str(beta_hat) + 
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' + 
          ', E[\hat{\Lambda}]=' + str([list(lmd_d) for lmd_d in np.round(E_lambda_hat_dd, 5)]) + 
          '$', loc='left')
#plt.xlim((mu_truth_d[0] - 0.5 * sigma_truth_dd[0, 0], mu_truth_d[0] + 0.5 * sigma_truth_dd[0, 0]))
#plt.ylim((mu_truth_d[1] - 0.5 * sigma_truth_dd[1, 1], mu_truth_d[1] + 0.5 * sigma_truth_dd[1, 1]))
plt.colorbar()
plt.show()

#%%

# muの期待値を計算:式(2.76)
E_mu_hat_d = m_hat_d

# 事後分布の期待値を用いた分布を計算:式(2.72)
posterior_lambda = multivariate_normal.pdf(
    x=x_point_arr, mean=E_mu_hat_d, cov=np.linalg.inv(E_lambda_hat_dd)
)
print(posterior_lambda)

#%%

# 事後分布の期待値を用いた分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, posterior_lambda.reshape(x_dims)) # 事後分布の期待を用いた分布
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 尤度
#plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1]) # 観測データ
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
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

# 予測分布のパラメータを計算:式(3.140')
mu_s_d = m_hat_d
lambda_s_hat_dd = (1.0 - D + nu_hat) * beta_hat / (1 + beta_hat) * w_hat_dd
nu_s_hat = 1.0 - D + nu_hat
print(mu_s_d)
print(lambda_s_hat_dd)
print(nu_s_hat)

# 予測分布を計算:式(3.121)
predict = multivariate_t.pdf(
    x=x_point_arr, loc=mu_s_d, shape=np.linalg.inv(lambda_s_hat_dd), df=nu_s_hat
)
print(predict)

#%%

# 予測分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, predict.reshape(x_dims)) # 予測分布
plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
            alpha=0.5, linestyles='--') # 尤度
#plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1]) # 観測データ
plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
plt.ylabel('$x_2$')
plt.suptitle("Multivariate Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\\nu}_s=' + str(nu_s_hat) + 
          ', \hat{\mu}_s=[' + ', '.join([str(mu) for mu in np.round(mu_s_d, 1)]) + ']' + 
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
mu_truth_d = np.array([25.0, 50.0])
sigma_truth_dd = np.array([[20.0, 15.0], [15.0, 30.0]])
lambda_truth_dd = np.linalg.inv(sigma_truth_dd**2)

# muの事前分布のパラメータを指定
beta = 1
m_d = np.array([0.0, 0.0])

# lambdaの事前分布のパラメータを指定
w_dd = np.array([[0.0005, 0], [0, 0.0005]])
inv_w_dd = np.linalg.inv(w_dd)
nu = 2

# lambdaの期待値を計算:式(2.89)
E_lambda_dd = nu * w_dd

# 初期値による予測分布のパラメータを計算:式(3.140)
mu_s_d = m_d
lambda_s_dd = (nu - 1.0) * beta / (1 + beta) * w_dd
nu_s = nu - 1.0

# データ数(試行回数)を指定
N = 100

# 作図用のmuの点を作成
mu_1_point = np.linspace(mu_truth_d[0] - 100.0, mu_truth_d[0] + 100.0, num=1000)
mu_2_point = np.linspace(mu_truth_d[1] - 100.0, mu_truth_d[1] + 100.0, num=1000)
mu_1_grid, mu_2_grid = np.meshgrid(mu_1_point, mu_2_point)
mu_point_arr = np.stack([mu_1_grid.flatten(), mu_2_grid.flatten()], axis=1)
mu_dims = mu_1_grid.shape

# 作図用のxの点を作成
x_1_point = np.linspace(mu_truth_d[0] - 4 * sigma_truth_dd[0, 0], mu_truth_d[0] + 4 * sigma_truth_dd[0, 0], num=300)
x_2_point = np.linspace(mu_truth_d[1] - 4 * sigma_truth_dd[1, 1], mu_truth_d[1] + 4 * sigma_truth_dd[1, 1], num=300)
x_1_grid, x_2_grid = np.meshgrid(x_1_point, x_2_point)
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape

# 推移の記録用の受け皿を初期化
x_nd = np.empty((N, 2))
trace_beta = [beta]
trace_m = [list(m_d)]
trace_posterior = [
    multivariate_normal.pdf(
        x=mu_point_arr, mean=m_d, cov=np.linalg.inv(beta * E_lambda_dd)
    )
]
trace_mu_s = [list(mu_s_d)]
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
    
    # lambdaの期待値を計算:式(2.89)
    E_lambda_dd = nu * np.linalg.inv(inv_w_dd)
    
    # muの事後分布を計算:式(2.72)
    trace_posterior.append(
        multivariate_normal.pdf(
            x=mu_point_arr, mean=m_d, cov=np.linalg.inv(beta * E_lambda_dd)
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
    
    # 超パラメータを記録
    trace_beta.append(beta)
    trace_m.append(list(m_d))
    trace_mu_s.append(list(mu_s_d))
    trace_lambda_s.append([list(lmd_d) for lmd_d in lambda_s_dd])
    trace_nu_s.append(nu_s)
    
    # 動作確認
    print('n=' + str(n + 1) + ' (' + str(np.round((n + 1) / N * 100, 1)) + '%)')

# 観測データを確認
print(x_nd[:5])

#%%

## muの事後分布の推移をgif画像化

## 画像サイズを指定
fig = plt.figure(figsize=(10, 7.5))

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
              ', \hat{\\beta}=' + str(trace_beta[n]) + 
              ', \hat{m}=[' + ', '.join([str(m) for m in np.round(trace_m[n], 1)]) + ']' + 
              '$', loc='left')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior_mu, frames=N + 1, interval=100)
posterior_anime.save("ch3_4_3_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.72)
true_model = multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_truth_dd)
)

# 画像サイズを指定
fig = plt.figure(figsize=(10, 7.5))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.contour(x_1_grid, x_2_grid, np.array(trace_predict[n]).reshape(x_dims)) # 予測分布
    plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dims), 
                alpha=0.5, linestyles='--') # 尤度
    plt.scatter(x=x_nd[:n, 0], y=x_nd[:n, 1]) # 観測データ
    plt.scatter(x=mu_truth_d[0], y=mu_truth_d[1], color='red', s=100, marker='x') # 真のmu
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.suptitle("Multivariate Student's t Distribution", fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\\nu}_s=' + str(trace_nu_s[n]) + 
              ', \hat{\mu}_s=[' + ', '.join([str(mu) for mu in np.round(trace_mu_s[n], 1)]) + ']' + 
              ', \hat{\Lambda}_s=' + str([list(lmd_d) for lmd_d in np.round(trace_lambda_s[n], 5)]) + 
              '$', loc='left')

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_4_3_Predict.gif")

#%%

print('end')

