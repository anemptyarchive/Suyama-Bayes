# 3.2.1 ベルヌーイ分布の学習と予測

#%%

# 3.2.1項で利用するライブラリ
import numpy as np
import math # 対数ガンマ関数:lgamma()
import matplotlib.pyplot as plt

#%%

## 真のモデルの設定

# 真のパラメータを指定
mu_true = 0.4

#%%

# x軸の値を設定
x_line = np.array([0, 1])

# 観測モデル(ベルヌーイ分布)を計算
true_model = np.array([1 - mu_true, mu_true])

# 観測モデルを作図
plt.bar(x=x_line, height=true_model) # 真のモデル
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Observation Model', fontsize=20)
plt.title('$\mu=' + str(mu_true) + '$', loc='left')
plt.show()

#%%

## 観測データの生成

# データ数を指定
N = 50

# (観測)データを生成
x_n = np.random.binomial(n=1, p=mu_true, size=N)

#%%

# 観測データのヒストグラムを作図
plt.bar(x=x_line, height=[N - np.sum(x_n), np.sum(x_n)]) # 観測データ
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu_true) + '$', loc='left')
plt.show()

#%%

## 事前分布の設定

# 事前分布のパラメータを指定
a = 1
b = 1

#%%

# x軸の値を設定
mu_line = np.arange(0.0, 1.01, 0.01)

# 事前分布(ベータ分布)を計算
ln_beta_C = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) # 正規化項
piror = np.exp(
    ln_beta_C + (a - 1) * np.log(mu_line) + (b - 1) * np.log(1 - mu_line)
)

# 事前分布を作図
plt.plot(mu_line, piror)
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Piror Distribution', fontsize=20)
plt.title('a=' + str(a) + ', b=' + str(b), loc='left')
plt.show()

#%%

## 事後分布の計算

# 事後分布のパラメータを計算
a_hat = np.sum(x_n) + a
b_hat = N - np.sum(x_n) + b

#%%

# 事後分布(ベータ分布)の確率密度を計算
ln_beta_C = math.lgamma(a_hat + b_hat) - math.lgamma(a_hat) - math.lgamma(b_hat)
posterior = np.exp(
    ln_beta_C + (a_hat - 1) * np.log(mu_line) + (b_hat - 1) * np.log(1 - mu_line)
)

# 事後分布を作図
plt.plot(mu_line, posterior) # 事後分布
plt.vlines(x=mu_true, ymin=0, ymax=max(posterior), linestyles='--', color='red') # 真のパラメータ
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Posterior Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(b_hat) + '$', loc='left')
plt.show()

#%%

## 予測分布の計算

# 予測分布のパラメータを計算
mu_hat_star = a_hat / (a_hat + b_hat)
mu_hat_star = (np.sum(x_n) + a) / (N + a + b)

#%%

# 予測分布(ベルヌーイ分布)を計算
predict = np.array([1 - mu_hat_star, mu_hat_star])

# 予測分布を作図
plt.bar(x=x_line, height=true_model, alpha=0.5, color='white', edgecolor='red', linestyle='dashed', label='true') # 真のモデル
plt.bar(x=x_line, height=predict, alpha=0.5, color='purple', label='predict') # 予測分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Predictive Distribution', fontsize=20)
plt.title('$\hat{\mu}_{*}=' + str(np.round(mu_hat_star, 2)) + '$', loc='left')
plt.legend()
plt.show()

#%%

## アニメーション

# 追加ライブラリ
import matplotlib.animation as animation

#%%

## ベイズ推論

# 真のパラメータを指定
mu_true = 0.4

# 事前分布のパラメータを指定
a = 1
b = 1

# データ数を指定
N = 100

# x軸の値を設定
mu_line = np.arange(0.0, 1.005, 0.005)

# 受け皿を初期化
x_n = np.empty(N)
trace_a = []
trace_b = []
trace_mu = []
trace_posterior = []
trace_predict = []

# 推論処理
for n in range(N):
    # (観測)データを生成
    x_n[n] = np.random.binomial(n=1, p=mu_true, size=1)
    
    # 事後分布のパラメータを計算
    a += x_n[n]
    b += 1 - x_n[n]
    
    # 事後分布を計算
    tmp_C = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) # 正規化項
    trace_posterior.append(
        np.exp(tmp_C + (a - 1) * np.log(mu_line) + (b - 1) * np.log(1 - mu_line))
    )
    
    # 予測分布のパラメータを計算
    mu_star = a / (a + b)
    
    # 予測分布を計算
    trace_predict.append(
        [1 - mu_star, mu_star]
    )
    
    # 値を記録
    trace_a.append(a)
    trace_b.append(b)
    trace_mu.append(mu_star)

#%%

## 予測分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 8))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の事後分布を作図
    plt.plot(mu_line, trace_posterior[n]) # 事後分布
    plt.bar(x=[0, 1], height=[(n - np.sum(x_n[:n])) / n, np.sum(x_n[:n]) / n], width=0.1)
    plt.vlines(x=mu_true, ymin=0, ymax=np.nanmax(trace_posterior), linestyles='--', color='red') # 真のパラメータ
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.suptitle('Posterior Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{a}=' + str(trace_a[n]) + ', \hat{b}=' + str(trace_b[n]) + '$', loc='left')
    plt.show()

# gif画像を作成
ani = animation.FuncAnimation(fig, update_posterior, frames=N, interval=100)
ani.save("figname.gif")

#%%

## 予測分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 8))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.bar(x=[0, 1], height=[1 - mu_true, mu_true], alpha=0.5, 
            color='white', edgecolor='red', linestyle='dashed', label='true') # 真のモデル
    plt.bar(x=[0, 1], height=trace_predict[n], alpha=0.5, color='purple', label='predict') # 予測分布
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.suptitle('Predictive Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\mu}_{*}=' + str(np.round(trace_mu[n], 2)) + '$', loc='left')
    plt.ylim((0, 1))
    plt.legend()
    plt.show()

# gif画像を作成
ani = animation.FuncAnimation(fig, update_predict, frames=N, interval=100)
ani.save("figname.gif")

#%%

print('end')
