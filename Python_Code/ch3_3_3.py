# 3.3.3 1次元ガウス分布の学習と予測：平均と精度が未知の場合

#%%

# 3.3.3項で利用するライブラリ
import numpy as np
import math # 対数ガンマ関数:lgamma()
from scipy.stats import norm, gamma, t # ガウス分布, ガンマ分布, スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真のパラメータを指定
mu_truth = 25
lambda_truth = 0.01
print(np.sqrt(1 / lambda_truth)) # 標準偏差

# 作図用のxの値を設定
x_line = np.arange(
    mu_truth - 4 * np.sqrt(1 / lambda_truth), 
    mu_truth + 4 * np.sqrt(1 / lambda_truth), 
    0.1
)

# 尤度を計算:式(2.64)
C_N = 1 / np.sqrt(2 * np.pi / lambda_truth) # 正規化項
true_model = C_N * np.exp(- 0.5 * lambda_truth * (x_line - mu_truth)**2) # 確率密度

# 尤度を計算:SciPy ver
true_model = norm.pdf(x=x_line, loc=mu_truth, scale=np.sqrt(1 / lambda_truth)) # 確率密度

#%%

# 尤度を作図
plt.plot(x_line, true_model, label='true model', color='purple') # 尤度
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$\mu=' + str(mu_truth) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend()
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# ガウス分布に従うデータを生成
x_n = np.random.normal(loc=mu_truth, scale=np.sqrt(1 / lambda_truth), size=N)
print(x_n[:5])

#%%

# 観測データのヒストグラムを作図
plt.hist(x=x_n, bins=50) # 観測データ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu_truth) + 
          ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.show()

#%%

## 事前分布(ガウス・ガンマ分布)の設定

# muの事前分布のパラメータを指定
m = 0
beta = 1

# lambdaの事前分布のパラメータを指定
a = 1
b = 1

# lambdaの期待値を計算:式(2.59)
E_lambda = a / b


# 作図用のmuの値を設定
mu_line = np.arange(mu_truth - 30, mu_truth + 30, 0.1)

# muの事前分布を計算:式(2.64)
C_N = 1 / np.sqrt(2 * np.pi / beta / E_lambda) # 正規化項
prior_mu = C_N * np.exp(- 0.5 * beta * E_lambda * (mu_line - m)**2) # 確率密度

# muの事前分布を計算:SciPy ver
#prior_mu = norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1 / beta / E_lambda)) # 確率密度


# 作図用のlambdaの値を設定
lambda_line = np.arange(0, 4 * lambda_truth, 0.00001)

# lambdaの事前分布を計算:式(2.56)
ln_C_Gam = a * np.log(b) - math.lgamma(a) # 正規化項(対数)
prior_lambda = np.exp(ln_C_Gam + (a - 1) * np.log(lambda_line) - b * lambda_line) # 確率密度

# lambdaの事前分布を計算:SciPy ver
#prior_lambda = gamma.pdf(x=lambda_line, a=a, scale=1 / b) # 確率密度

#%%

# muの事前分布を作図
plt.plot(mu_line, prior_mu, label='$\mu$ prior', color='purple') # muの事前分布
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$m=' + str(m) + ', \\beta=' + str(beta) + '$', loc='left')
plt.legend()
plt.show()

#%%

# lambdaの事前分布を作図
plt.plot(lambda_line, prior_lambda, label='$\lambda$ prior', color='purple') # lambdaの事前分布
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + '$', loc='left')
plt.legend()
plt.show()

#%%

## 事後分布(ガウス・ガンマ分布)の計算

# muの事後分布のパラメータを計算:式(3.83)
beta_hat = N + beta
m_hat = (np.sum(x_n) + beta * m) / beta_hat
print(beta_hat)
print(m_hat)

# lambdaの事後分布のパラメータを計算:式(3.88)
a_hat = 0.5 * N + a
b_hat = 0.5 * (np.sum(x_n**2) + beta * m**2 - beta_hat * m_hat**2) + b
print(a_hat)
print(b_hat)

# lambdaの期待値を計算:式(2.59)
E_lambda_hat = a_hat / b_hat
print(E_lambda_hat)

# muの事後分布を計算:式(2.64)
C_N = 1 / np.sqrt(2 * np.pi / beta_hat / E_lambda_hat) # 正規化項
posterior_mu = C_N * np.exp(- 0.5 * beta_hat * E_lambda_hat * (mu_line - m_hat)**2) # 確率密度

# muの事前分布を計算:SciPy ver
#posterior_mu = norm.pdf(x=mu_line, loc=m_hat, scale=np.sqrt(1 / beta_hat / E_lambda_hat)) # 確率密度

# lambdaの事後分布の計算:式(2.56)
ln_C_Gam = a_hat * np.log(b_hat) - math.lgamma(a_hat) # 正規化項(対数)
posterior_lambda = np.exp(ln_C_Gam + (a_hat - 1) * np.log(lambda_line) - b_hat * lambda_line) # 確率密度

# lambdaの事前分布を計算:SciPy ver
#posterior_lambda = gamma.pdf(x=lambda_line, a=a_hat, scale=1 / b_hat) # 確率密度

#%%

# muの事前分布を作図
plt.plot(mu_line, posterior_mu, label='$\mu$ posterior', color='purple') # muの事後分布
plt.vlines(x=mu_truth, ymin=0, ymax=max(posterior_mu), 
           label='$\mu$ truth', color='red', linestyle='--') # 真のmu
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$\hat{m}=' + str(np.round(m_hat, 1)) + 
          ', \hat{\\beta}=' + str(beta_hat) + '$', loc='left')
plt.legend()
plt.show()

#%%

# lambdaの事後分布を作図
plt.plot(lambda_line, posterior_lambda, label='$\lambda$ posterior', color='purple') # lambdaの事後分布
plt.vlines(x=lambda_truth, ymin=0, ymax=max(posterior_lambda), label='$\lambda$ truth', 
           color='red', linestyle='--') # 真のlambda
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', a=' + str(a_hat) + ', b=' + str(np.round(b_hat, 1)) + '$', loc='left')
plt.legend()
plt.show()

#%%

## 予測分布(スチューデントのt分布)の計算

# 予測分布のパラメータを計算:式(3.95')
mu_s_hat = m_hat
lambda_s_hat = beta_hat * a_hat / (1 + beta_hat) / b_hat
nu_s_hat = 2 * a_hat

# 予測分布を計算:式(3.76)
ln_C_St = math.lgamma(0.5 * (nu_s_hat + 1)) - math.lgamma(0.5 * nu_s_hat) # 正規化項(対数)
ln_term1 = 0.5 * np.log(lambda_s_hat / np.pi / nu_s_hat)
ln_term2 = - 0.5 * (nu_s_hat + 1) * np.log(1 + lambda_s_hat / nu_s_hat * (x_line - mu_s_hat)**2)
predict = np.exp(ln_C_St + ln_term1 + ln_term2) # 確率密度

# 予測分布を計算:SciPy ver
predict = t.pdf(x=x_line, df=nu_s_hat, loc=mu_s_hat, scale=np.sqrt(1 / lambda_s_hat))

#%%

# 予測分布を作図
plt.plot(x_line, predict, label='predict', color='purple') # 予測分布
plt.plot(x_line, true_model, label='true', color='red', linestyle='--') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle("Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\mu}_s=' + str(np.round(mu_s_hat, 1)) + 
          ', \hat{\lambda}_s=' + str(np.round(lambda_s_hat, 3)) + 
          ', \hat{\\nu}_s=' + str(nu_s_hat) + '$', loc='left')
plt.legend()
plt.show()

#%%

### ・アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import norm, gamma, t # ガウス分布, ガンマ分布, スチューデントのt分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## 推論処理

# 真のパラメータを指定
mu_truth = 25
lambda_truth = 0.01

# muの事前分布のパラメータを指定
m = 0
beta = 1

# lambdaの事前分布のパラメータを指定
a = 1
b = 1

# lambdaの期待値を計算:式(2.59)
E_lambda = a / b

# 初期値による予測分布のパラメータを計算:式(3.95)
mu_s = m
lambda_s = beta * a / (1 + beta) / b
nu_s = 2 * a

# データ数(試行回数)を指定
N = 100

# 作図用の値を設定
x_line = np.arange(
    mu_truth - 4 * np.sqrt(1 / lambda_truth), 
    mu_truth + 4 * np.sqrt(1 / lambda_truth), 
    0.1
)
mu_line = np.arange(mu_truth - 30, mu_truth + 30, 0.1)
lambda_line = np.arange(0, 4 * lambda_truth, 0.00001)

# 推移の記録用の受け皿を初期化
x_n = np.empty(N)
trace_m = [m]
trace_beta = [beta]
trace_posterior_mu = [norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1 / beta / E_lambda))]
trace_a = [a]
trace_b = [b]
trace_posterior_lambda = [gamma.pdf(x=lambda_line, a=a, scale=1 / b)]
trace_mu_s = [mu_s]
trace_lambda_s = [lambda_s]
trace_nu_s = [nu_s]
trace_predict = [t.pdf(x=x_line, df=nu_s, loc=mu_s, scale=np.sqrt(1 / lambda_s))]

# ベイズ推論
for n in range(N):
    # ガウス分布に従うデータを生成
    x_n[n] = np.random.normal(loc=mu_truth, scale=np.sqrt(1 / lambda_truth), size=1)
    
    # muの事後分布のパラメータを更新:式(3.83)
    old_beta = beta
    old_m = m
    beta += 1
    m = (x_n[n] + old_beta * m) / beta
    
    # lambdaの事後分布のパラメータを更新:式(3.88)
    a += 0.5
    b += 0.5 * (x_n[n]**2 + old_beta * old_m**2 - beta * m**2)
    
    # lambdaの期待値を計算:式(2.59)
    E_lambda = a / b
    
    # muの事前分布を計算:式(2.64)
    trace_posterior_mu.append(
        norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1 / beta / E_lambda))
    )
    
    # lambdaの事前分布を計算:式(2.56)
    trace_posterior_lambda.append(gamma.pdf(x=lambda_line, a=a, scale=1 / b))
    
    # 予測分布のパラメータを更新:式(3.95)
    mu_s = m
    lambda_s = beta * a / (1 + beta) / b
    nu_s = 2 * a
    
    # 予測分布を計算:式(3.76)
    trace_predict.append(t.pdf(x=x_line, df=nu_s, loc=mu_s, scale=np.sqrt(1 / lambda_s)))
    
    # 超パラメータを記録
    trace_m.append(m)
    trace_beta.append(beta)
    trace_a.append(a)
    trace_b.append(b)
    trace_mu_s.append(mu_s)
    trace_lambda_s.append(lambda_s)
    trace_nu_s.append(nu_s)

#%%

## muの事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior_mu(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目のmuの事後分布を作図
    plt.plot(mu_line, trace_posterior_mu[n], label='$\mu$ posterior', color='purple') # muの事後分布
    plt.vlines(x=mu_truth, ymin=0, ymax=np.nanmax(trace_posterior_mu), 
               label='$\mu$ trurh', color='red', linestyle='--') # 真のmu
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.suptitle('Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{m}=' + str(np.round(trace_m[n], 1)) + 
              ', \hat{\\beta}=' + str(trace_beta[n]) + '$', loc='left')
    plt.legend()

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior_mu, frames=N + 1, interval=100)
posterior_anime.save("ch3_3_3_Posterior_mu.gif")

#%%

## lambdaの事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior_lambda(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目のlambdaの事後分布を作図
    plt.plot(lambda_line, trace_posterior_lambda[n], 
             label='$\lambda$ posterior', color='purple') # lambdaの事後分布
    plt.vlines(x=lambda_truth, ymin=0, ymax=np.nanmax(trace_posterior_lambda), 
               label='$\lambda$ trurh', color='red', linestyle='--') # 真のlambda
    plt.xlabel('$\lambda$')
    plt.ylabel('density')
    plt.suptitle('Gamma Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{a}=' + str(trace_a[n]) + 
              ', \hat{b}=' + str(np.round(trace_b[n], 1)) + '$', loc='left')
    plt.legend()

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior_lambda, frames=N + 1, interval=100)
posterior_anime.save("ch3_3_3_Posterior_lambda.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.64)
true_model = norm.pdf(x=x_line, loc=mu_truth, scale=np.sqrt(1 / lambda_truth)) # 確率密度

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
    plt.title('$N=' + str(n) + ', \hat{\mu}_s=' + str(np.round(trace_mu_s[n], 1)) + 
              ', \hat{\lambda}_s=' + str(np.round(trace_lambda_s[n], 3)) + 
              ', \hat{\\nu}_s=' + str(trace_nu_s[n]) + '$', loc='left')
    plt.ylim(0, np.nanmax(trace_predict))
    plt.legend()

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_3_3_Predict.gif")

#%%

print('end')

