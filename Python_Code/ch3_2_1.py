# 3.2.1 ベルヌーイ分布の学習と予測

#%%

# 3.2.1項で利用するライブラリ
import numpy as np
import math # 対数ガンマ関数:lgamma()
#from scipy.stats import beta # ベータ分布
import matplotlib.pyplot as plt

#%%

## 真のモデルの設定

# 真のパラメータを指定
mu_truth = 0.25

# x軸の値を設定
x_point = np.array([0, 1])

# 尤度(ベルヌーイ分布)を計算
true_model = np.array([1 - mu_truth, mu_truth]) # 確率
print(true_model)

#%%

# 尤度を作図
plt.bar(x=x_point, height=true_model, color='purple') # 尤度
plt.xlabel('x')
plt.ylabel('prob')
plt.xticks(ticks=x_point, labels=x_point) # x軸目盛
plt.suptitle('Bernoulli Distribution', fontsize=20)
plt.title('$\mu=' + str(mu_truth) + '$', loc='left')
plt.ylim(0.0, 1.0)
plt.show()

#%%

## 観測データの生成

# データ数を指定
N = 50

# (観測)データを生成
x_n = np.random.binomial(n=1, p=mu_truth, size=N)
print(np.sum(x_n) / N)

#%%

# 観測データのヒストグラムを作図
plt.bar(x=x_point, height=[N - np.sum(x_n), np.sum(x_n)]) # 観測データ
plt.xlabel('x')
plt.ylabel('count')
plt.xticks(ticks=x_point, labels=x_point) # x軸目盛
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu_truth) + '$', loc='left')
plt.show()

#%%

## 事前分布の設定

# 事前分布のパラメータを指定
a = 1.0
b = 1.0

# x軸の値を設定
mu_line = np.arange(0.0, 1.001, 0.001)

# 事前分布(ベータ分布)を計算
ln_C_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) # 正規化項(対数)
prior = np.exp(ln_C_beta) * mu_line**(a - 1) * (1 - mu_line)**(b - 1) # 確率密度
#prior = beta.pdf(x=mu_line, a=a, b=b) # 確率密度:SciPy ver

#%%

# 事前分布を作図
plt.plot(mu_line, prior, color='purple')
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Beta Distribution', fontsize=20)
plt.title('a=' + str(a) + ', b=' + str(b), loc='left')
plt.show()

#%%

## 事後分布の計算

# 事後分布のパラメータを計算
a_hat = np.sum(x_n) + a
b_hat = N - np.sum(x_n) + b
print(a_hat)
print(b_hat)

# 事後分布(ベータ分布)の確率密度を計算
ln_C_beta = math.lgamma(a_hat + b_hat) - math.lgamma(a_hat) - math.lgamma(b_hat) # 正規化項(対数)
posterior = np.exp(ln_C_beta) * mu_line**(a_hat - 1) * (1 - mu_line)**(b_hat - 1) # 確率密度
#posterior = beta.pdf(x=mu_line, a=a_hat, b=b_hat) # 確率密度:SciPy ver

#%%

# 事後分布を作図
plt.plot(mu_line, posterior, color='purple') # 事後分布
plt.vlines(x=mu_truth, ymin=0, ymax=max(posterior), linestyles='--', color='red') # 真のパラメータ
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Beta Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(b_hat) + '$', loc='left')
plt.show()

#%%

## 予測分布の計算

# 予測分布のパラメータを計算
mu_hat_star = a_hat / (a_hat + b_hat)
#mu_hat_star = (np.sum(x_n) + a) / (N + a + b)
print(mu_hat_star)

# 予測分布(ベルヌーイ分布)を計算
predict = np.array([1 - mu_hat_star, mu_hat_star]) # 確率

#%%

# 予測分布を作図
plt.bar(x=x_point, height=true_model, label='true', alpha=0.5, 
        color='white', edgecolor='red', linestyle='dashed') # 真のモデル
plt.bar(x=x_point, height=predict, label='predict', alpha=0.5, 
        color='purple') # 予測分布
plt.xlabel('x')
plt.ylabel('prob')
plt.xticks(ticks=x_point, labels=x_point) # x軸目盛
plt.suptitle('Bernoulli Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\mu}_{*}=' + str(np.round(mu_hat_star, 2)) + '$', loc='left')
plt.ylim(0.0, 1.0)
plt.legend()
plt.show()


#%%

## アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import beta # ベータ分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## ベイズ推論

# 真のパラメータを指定
mu_truth = 0.4

# 事前分布のパラメータを指定
a = 1.0
b = 1.0

# 初期値による予測分布のパラメーターを計算
mu_star = a / (a + b)

# 事前分布のx軸の値を作成
mu_line = np.arange(0.0, 1.001, 0.001)

# データ数(試行回数)を指定
N = 100

# 推移の記録用の受け皿を初期化
x_n = np.empty(N)
trace_a = [a]
trace_b = [b]
trace_mu = [mu_star]
trace_posterior = [beta.pdf(x=mu_line, a=a, b=b)]
trace_predict = [[1 - mu_star, mu_star]]

# 推論処理
for n in range(N):
    # (観測)データを生成
    x_n[n] = np.random.binomial(n=1, p=mu_truth, size=1)
    
    # 事後分布のパラメータを計算
    a += x_n[n]
    b += 1 - x_n[n]
    
    # 事後分布を計算
    trace_posterior.append(beta.pdf(x=mu_line, a=a, b=b))
    
    # 予測分布のパラメータを計算
    mu_star = a / (a + b)
    
    # 予測分布を計算
    trace_predict.append([1 - mu_star, mu_star])
    
    # 値を記録
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
    
    # nフレーム目の事後分布を作図
    plt.plot(mu_line, trace_posterior[n], color='purple') # 事後分布
    plt.vlines(x=mu_truth, ymin=0, ymax=np.nanmax(trace_posterior), 
               linestyles='--', color='red') # 真のパラメータ
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.suptitle('Beta Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{a}=' + str(trace_a[n]) + ', \hat{b}=' + str(trace_b[n]) + '$', loc='left')

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_2_1_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# x軸の点を作成
x_point = np.array([0, 1])

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.bar(x=x_point, height=[1 - mu_truth, mu_truth], alpha=0.5, 
            color='white', edgecolor='red', linestyle='dashed', label='true') # 真のモデル
    plt.bar(x=x_point, height=trace_predict[n], alpha=0.5, color='purple', label='predict') # 予測分布
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.xticks(ticks=x_point, labels=x_point) # x軸目盛
    plt.suptitle('Bernoulli Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\mu}_{*}=' + str(np.round(trace_mu[n], 2)) + '$', loc='left')
    plt.ylim(0.0, 1.0)
    plt.legend()

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_2_1_Predict.gif")

#%%

print('end')
