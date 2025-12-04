
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


#%%

### ・アニメーションによる推移の確認

# 3.3.1項で利用するライブラリ
import numpy as np
from scipy.stats import norm # 1次元ガウス分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# 真の平均パラメータを指定
mu_truth = 25.0

# (既知の)精度パラメータを指定
lmd = 0.01


# muの事前分布の平均パラメータを指定
m = 0

# muの事前分布の精度パラメータを指定
lambda_mu = 0.001


# 初期値による予測分布のパラメータを計算:式(3.62)
mu_star = m
lambda_star = lmd * lambda_mu / (lmd + lambda_mu)


# 作図用のmuの値を設定
mu_line = np.linspace(
    mu_truth - 2.0 * np.sqrt(1.0 / lambda_mu), 
    mu_truth + 2.0 * np.sqrt(1.0 / lambda_mu), 
    num=1000
)

# 作図用のxの値を設定
x_line = np.linspace(
    mu_truth - 4 * np.sqrt(1 / lmd), 
    mu_truth + 4 * np.sqrt(1 / lmd), 
    num=1000
)

#%%

## 推論処理

# データ数(試行回数)を指定
N = 100

# 観測データの受け皿を作成
x_n = np.empty(N)

# 推移の記録用の受け皿を初期化
trace_m = [m]
trace_lambda_mu = [lambda_mu]
trace_posterior = [norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1.0 / lambda_mu))]
trace_mu_star = [mu_star]
trace_lambda_star = [lambda_star]
trace_predict = [norm.pdf(x=x_line, loc=mu_star, scale=np.sqrt(1.0 / lambda_star))]

# ベイズ推論
for n in range(N):
    # ガウス分布に従うデータを生成
    x_n[n] = np.random.normal(loc=mu_truth, scale=np.sqrt(1 / lmd), size=1)
    
    # muの事後分布のパラメータを計算:式(3.53),(3.54)
    lambda_mu_old = lambda_mu
    lambda_mu += lmd
    m = (lmd * x_n[n] + lambda_mu_old * m) / lambda_mu
    
    # muの事後分布(ガウス分布)を計算:式(2.64)
    trace_posterior.append(
        norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1.0 / lambda_mu))
    )
    
    # 予測分布のパラメータを計算:式(3.62)
    mu_star = m
    lambda_star = lmd * lambda_mu / (lmd + lambda_mu)
    
    # 予測分布(ガウス分布)を計算:式(2.64)
    trace_predict.append(
        norm.pdf(x=x_line, loc=mu_star, scale=np.sqrt(1.0 / lambda_star))
    )
    
    # n回目の結果を記録
    trace_m.append(m)
    trace_lambda_mu.append(lambda_mu)
    trace_mu_star.append(mu_star)
    trace_lambda_star.append(lambda_star)

#%%

## 事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目のmuの事後分布を作図
    plt.plot(mu_line, trace_posterior[n], label='posterior', color='purple') # muの事後分布
    plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(trace_posterior), 
               color='red', linestyle='--', label='true val') # 真の値
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[:n-1], y=np.repeat(0.0, n - 1), label='data') # 観測データ
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.suptitle('Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{m}=' + str(np.round(trace_m[n], 1)) + 
              ', \hat{\lambda}_{\mu}=' + str(np.round(trace_lambda_mu[n], 5)) + '$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_3_1_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度の確率密度を計算:式(2.64)
true_model = norm.pdf(x=x_line, loc=mu_truth, scale=np.sqrt(1.0 / lmd))

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の予測分布を作図
    plt.plot(x_line, true_model, color='red', linestyle='--', label='true model') # 真の分布
    plt.plot(x_line, trace_predict[n], color='purple', label='predict') # 予測分布
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[:n-1], y=np.repeat(0.0, n - 1), label='data') # 観測データ
    plt.xlabel('x')
    plt.ylabel('density')
    plt.suptitle('Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\mu}_{*}=' + str(np.round(trace_mu_star[n], 1)) + 
              ', \hat{\lambda}_{*}=' + str(np.round(trace_lambda_star[n], 3)) + '$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_3_1_Predict.gif")

#%%

print('end')

