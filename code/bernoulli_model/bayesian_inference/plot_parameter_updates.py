
### アニメーションによる推移の確認

# 3.2.1項で利用するライブラリ
import numpy as np
from scipy.stats import beta # ベータ分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# 真のパラメータを指定
mu_truth = 0.4

# 事前分布のパラメータを指定
a = 1.0
b = 1.0

# 初期値による予測分布のパラメーターを計算
mu_star = a / (a + b)


# 事前分布のx軸の値を作成
mu_line = np.arange(0.0, 1.001, 0.001)

#%%

## 推論処理

# データ数(試行回数)を指定
N = 100

# 観測データの受け皿を作成
x_n = np.empty(N)

# 推移の記録用の受け皿を初期化
trace_a = [a]
trace_b = [b]
trace_posterior = [beta.pdf(x=mu_line, a=a, b=b)]
trace_mu = [mu_star]
trace_predict = [[1 - mu_star, mu_star]]

# ベイズ推論
for n in range(N):
    # (観測)データを生成
    x_n[n] = np.random.binomial(n=1, p=mu_truth, size=1)
    
    # 事後分布のパラメータを更新:式(3.15)
    a += x_n[n]
    b += 1 - x_n[n]
    
    # 事後分布(ベータ分布)の確率密度を計算:式(2.41)
    trace_posterior.append(beta.pdf(x=mu_line, a=a, b=b))
    
    # 予測分布のパラメータを更新:式(3.19)
    mu_star = a / (a + b)
    
    # 予測分布(ベルヌーイ分布)の確率を計算:式(2.16)
    trace_predict.append([1 - mu_star, mu_star])
    
    # n回目の結果を記録
    trace_a.append(a)
    trace_b.append(b)
    trace_mu.append(mu_star)

#%%

## 事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の事後分布を作図
    plt.plot(mu_line, trace_posterior[n], color='purple', label='posterior') # 事後分布
    plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(trace_posterior), 
               color='red', linestyles='--', label='true val') # 真の値
    if n > 0: # 初回は除く
        plt.scatter(x_n[n-1], y=0.0, s=100, label='data') # 観測データ
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.suptitle('Beta Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{a}=' + str(trace_a[n]) + 
              ', \hat{b}=' + str(trace_b[n]) + '$', 
              loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_2_1_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# x軸の点を作成
x_point = np.array([0.0, 1.0])

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の予測分布を作図
    plt.bar(x=x_point, height=[1.0 - mu_truth, mu_truth], 
            color='white', edgecolor='red', linestyle='--', label='true model', zorder=0) # 真の分布
    plt.bar(x=x_point, height=trace_predict[n], 
            alpha=0.6, color='purple', label='predict', zorder=1) # 予測分布
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[n-1], y=0.0, s=100, label='data', zorder=2) # 観測データ
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.xticks(ticks=x_point, labels=x_point) # x軸目盛
    plt.suptitle('Bernoulli Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\mu}_{*}=' + str(np.round(trace_mu[n], 5)) + '$', 
              loc='left')
    plt.legend() # 凡例
    plt.ylim(-0.01, 1.0) # y軸の表示範囲

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_2_1_Predict.gif")

#%%

print('end')

