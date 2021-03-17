# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合

#%%

# 3.3.2項で利用するライブラリ
import numpy as np
import math # 対数ガンマ関数:lgamma()
from scipy.stats import norm, gamma, t # 1次元ガウス分布, ガンマ分布, 1次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真のパラメータを指定
mu = 25
lambda_truth = 0.01
print(np.sqrt(1 / lambda_truth)) # 標準偏差

# 作図用のxの値を設定
x_line = np.linspace(
    mu - 4 * np.sqrt(1 / lambda_truth), 
    mu + 4 * np.sqrt(1 / lambda_truth), 
    num=1000
)

# 尤度を計算:式(2.64)
ln_C_N = - 0.5 * (np.log(2 * np.pi) - np.log(lambda_truth)) # 正規化項(対数)
true_model = np.exp(ln_C_N - 0.5 * lambda_truth * (x_line - mu)**2) # 確率密度

# 尤度を計算:SciPy ver
#true_model = norm.pdf(x=x_line, loc=mu, scale=np.sqrt(1 / lambda_truth)) # 確率密度

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, true_model, color='purple') # 尤度
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$\mu=' + str(mu) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# ガウス分布に従うデータを生成
x_n = np.random.normal(loc=mu, scale=np.sqrt(1 / lambda_truth), size=N)
print(x_n[:5])

#%%

# 観測データのヒストグラムを作図
plt.figure(figsize=(12, 9))
plt.hist(x=x_n, bins=50) # 観測データ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu) + 
          ', \sigma=' + str(np.sqrt(1 / lambda_truth)) + '$', loc='left')
plt.show()

#%%

## 事前分布(ガンマ分布)の設定

# lambdaの事前分布のパラメータを指定
a = 1
b = 1

# 作図用のlambdaの値を設定
lambda_line = np.linspace(0, 4 * lambda_truth, num=1000)

# lambdaの事前分布を計算:式(2.56)
ln_C_Gam = a * np.log(b) - math.lgamma(a) # 正規化項(対数)
prior = np.exp(ln_C_Gam + (a - 1) * np.log(lambda_line) - b * lambda_line) # 確率密度

# lambdaの事前分布を計算:SciPy ver
#prior = gamma.pdf(x=lambda_line, a=a, scale=1 / b) # 確率密度

#%%

# lambdaの事前分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, prior, label='prior', color='purple') # lambdaの事前分布
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + '$', loc='left')
plt.show()

#%%

## 事後分布(ガンマ分布)の計算

# lambdaの事後分布のパラメータを計算:式(3.69)
a_hat = 0.5 * N + a
b_hat = 0.5 * np.sum((x_n - mu)**2) + b
print(a_hat)
print(b_hat)

# lambdaの事後分布の計算:式(2.56)
ln_C_Gam = a_hat * np.log(b_hat) - math.lgamma(a_hat) # 正規化項(対数)
posterior = np.exp(ln_C_Gam + (a_hat - 1) * np.log(lambda_line) - b_hat * lambda_line) # 確率密度

# lambdaの事前分布を計算:SciPy ver
#posterior = gamma.pdf(x=lambda_line, a=a_hat, scale=1 / b_hat) # 確率密度

#%%

# lambdaの事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, posterior, label='posterior', color='purple') # lambdaの事後分布
plt.vlines(x=lambda_truth, ymin=0, ymax=max(posterior), label='$\lambda_{truth}$', 
           color='red', linestyle='--') # 真のlambda
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', a=' + str(a_hat) + ', b=' + str(np.round(b_hat, 1)) + '$', loc='left')
plt.legend()
plt.show()

#%%

## 予測分布(スチューデントのt分布)の計算

# 予測分布のパラメータを計算:式(3.79')
mu_s = mu
lambda_s_hat = a_hat / b_hat
nu_s_hat = 2 * a_hat
#lambda_s_hat = (N + 2 * a) / (np.sum((x_n - mu)**2) + 2 * b)
#nu_s_hat = N + 2 * a
print(mu_s)
print(lambda_s_hat)
print(nu_s_hat)

# 予測分布を計算:式(3.76)
ln_C_St = math.lgamma(0.5 * (nu_s_hat + 1)) - math.lgamma(0.5 * nu_s_hat) # 正規化項(対数)
ln_term1 = 0.5 * np.log(lambda_s_hat / np.pi / nu_s_hat)
ln_term2 = - 0.5 * (nu_s_hat + 1) * np.log(1 + lambda_s_hat / nu_s_hat * (x_line - mu_s)**2)
predict = np.exp(ln_C_St + ln_term1 + ln_term2) # 確率密度

# 予測分布を計算:SciPy ver
#predict = t.pdf(x=x_line, df=nu_s_hat, loc=mu_s, scale=np.sqrt(1 / lambda_s_hat))

#%%

# 予測分布を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, predict, label='predict', color='purple') # 予測分布
plt.plot(x_line, true_model, label='true', color='red', linestyle='--') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle("Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + ', \mu_s=' + str(mu_s) + 
          ', \hat{\lambda}_s=' + str(np.round(lambda_s_hat, 3)) + 
          ', \hat{\\nu}_s=' + str(nu_s_hat) + '$', loc='left')
plt.legend()
plt.show()

#%%

### ・アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import norm, gamma, t # 1次元ガウス分布, ガンマ分布, 1次元スチューデントのt分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## 推論処理

# 真のパラメータを指定
mu = 25
lambda_truth = 0.01

# lambdaの事前分布のパラメータを指定
a = 1
b = 1

# 初期値による予測分布のパラメータを計算:式(3.79)
mu_s = mu
lambda_s = a / b
nu_s = 2 * a

# 作図用のxの値を設定
x_line = np.linspace(
    mu - 4 * np.sqrt(1 / lambda_truth), 
    mu + 4 * np.sqrt(1 / lambda_truth), 
    num=1000
)

# 作図用のlambdaの値を設定
lambda_line = np.linspace(0, 4 * lambda_truth, num=1000)

# データ数(試行回数)を指定
N = 100

# 推移の記録用の受け皿を初期化
x_n = np.empty(N)
trace_a = [a]
trace_b = [b]
trace_posterior = [gamma.pdf(x=lambda_line, a=a, scale=1 / b)]
trace_mu_s = [mu_s]
trace_lambda_s = [lambda_s]
trace_nu_s = [nu_s]
trace_predict = [t.pdf(x=x_line, df=nu_s, loc=mu_s, scale=np.sqrt(1 / lambda_s))]

# ベイズ推論
for n in range(N):
    # ガウス分布に従うデータを生成
    x_n[n] = np.random.normal(loc=mu, scale=np.sqrt(1 / lambda_truth), size=1)
    
    # lambdaの事前分布のパラメータを更新:式(3.69)
    a += 0.5
    b += 0.5 * (x_n[n] - mu)**2
    
    # lambdaの事前分布(ガンマ分布)を計算:式(2.56)
    trace_posterior.append(gamma.pdf(x=lambda_line, a=a, scale=1 / b))
    
    # 予測分布のパラメータを更新:式(3.79)
    mu_s = mu
    lambda_s = a / b
    nu_s = 2 * a
    
    # 予測分布(スチューデントのt分布)を計算:式(3.76)
    trace_predict.append(
        t.pdf(x=x_line, df=nu_s, loc=mu_s, scale=np.sqrt(1 / lambda_s))
    )
    
    # 超パラメータを記録
    trace_a.append(a)
    trace_b.append(b)
    trace_mu_s.append(mu_s)
    trace_lambda_s.append(lambda_s)
    trace_nu_s.append(nu_s)

# 観測データを確認
print(x_n[:5])

#%%

## 事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目のlambdaの事後分布を作図
    plt.plot(lambda_line, trace_posterior[n], label='posterior', color='purple') # lambdaの事後分布
    plt.vlines(x=lambda_truth, ymin=0, ymax=np.nanmax(trace_posterior), 
               label='$\lambda_{trurh}$', color='red', linestyle='--') # 真のlambda
    plt.xlabel('$\lambda$')
    plt.ylabel('density')
    plt.suptitle('Gamma Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{a}=' + str(trace_a[n]) + 
              ', \hat{b}=' + str(np.round(trace_b[n], 1)) + '$', loc='left')
    plt.legend()

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_3_2_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.64)
true_model = norm.pdf(x=x_line, loc=mu, scale=np.sqrt(1 / lambda_truth)) # 確率密度

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.plot(x_line, trace_predict[n], label='predict', color='purple') # 予測分布
    plt.plot(x_line, true_model, label='true', color='red', linestyle='--') # 真の分布
    plt.xlabel('x')
    plt.ylabel('density')
    plt.suptitle("Student's t Distribution", fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\mu}_s=' + str(trace_mu_s[n]) + 
              ', \hat{\lambda}_s=' + str(np.round(trace_lambda_s[n], 3)) + 
              ', \hat{\\nu}_s=' + str(trace_nu_s[n]) + '$', loc='left')
    plt.ylim(0, np.nanmax(trace_predict))
    plt.legend()

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_3_2_Predict.gif")

#%%

print('end')

