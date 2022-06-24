# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合

#%%

# 3.3.2項で利用するライブラリ
import numpy as np
from scipy.special import gammaln # 対数ガンマ関数
from scipy.stats import norm, gamma, t # 1次元ガウス分布, ガンマ分布, 1次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# (既知の)平均パラメータを指定
mu = 25

# 真の精度パラメータを指定
lambda_truth = 0.01
print(np.sqrt(1 / lambda_truth)) # 標準偏差


# 作図用のxの値を設定
x_line = np.linspace(
    mu - 4.0 * np.sqrt(1.0 / lambda_truth), 
    mu + 4.0 * np.sqrt(1.0 / lambda_truth), 
    num=1000
)


# 尤度の確率密度を計算:式(2.64)
ln_C_N = - 0.5 * (np.log(2.0 * np.pi) - np.log(lambda_truth)) # 正規化項(対数)
model_dens = np.exp(ln_C_N - 0.5 * lambda_truth * (x_line - mu)**2)
#model_dens = norm.pdf(x=x_line, loc=mu, scale=np.sqrt(1.0 / lambda_truth))

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, model_dens, label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$\mu=' + str(mu) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# ガウス分布に従うデータを生成
x_n = np.random.normal(loc=mu, scale=np.sqrt(1.0 / lambda_truth), size=N)
print(x_n[:5])

#%%

# 観測データのヒストグラムを作成
plt.figure(figsize=(12, 9))
plt.plot(x_line, model_dens, color='red', linestyle='--', label='true model') # 真の分布
plt.hist(x=x_n, density=True, bins=50, label='data') # 観測データ:(相対度数)
#plt.hist(x=x_n, bins=50, label='data') # 観測データ:(度数)
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事前分布(ガンマ分布)の設定

# lambdaの事前分布のパラメータを指定
a = 1.0
b = 1.0

# 作図用のlambdaの値を設定
lambda_line = np.linspace(0.0, 4.0 * lambda_truth, num=1000)


# lambdaの事前分布の確率密度を計算:式(2.56)
ln_C_Gam = a * np.log(b) - gammaln(a) # 正規化項(対数)
prior_dens = np.exp(ln_C_Gam + (a - 1.0) * np.log(lambda_line) - b * lambda_line)
#prior_dens = gamma.pdf(x=lambda_line, a=a, scale=1.0 / b)

#%%

# lambdaの事前分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, prior_dens, color='purple', label='prior') # lambdaの事前分布
plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(prior_dens) * 2, 
           color='red', linestyle='--', label='ture val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事後分布(ガンマ分布)の計算

# lambdaの事後分布のパラメータを計算:式(3.69)
a_hat = 0.5 * N + a
b_hat = 0.5 * np.sum((x_n - mu)**2) + b


# lambdaの事後分布の確率密度を計算:式(2.56)
ln_C_Gam = a_hat * np.log(b_hat) - gammaln(a_hat) # 正規化項(対数)
posterior_dens = np.exp(ln_C_Gam + (a_hat - 1.0) * np.log(lambda_line) - b_hat * lambda_line)
#posterior_dens = gamma.pdf(x=lambda_line, a=a_hat, scale=1.0 / b_hat)

#%%

# lambdaの事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, posterior_dens, color='purple', label='posterior') # lambdaの事後分布
plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(posterior_dens), 
           color='red', linestyle='--', label='ture val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', a=' + str(a_hat) + ', b=' + str(np.round(b_hat, 1)) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 予測分布(スチューデントのt分布)の計算

# 予測分布のパラメータを計算:式(3.79')
mu_s = mu
lambda_s_hat = a_hat / b_hat
nu_s_hat = 2 * a_hat
#lambda_s_hat = (N + 2 * a) / (np.sum((x_n - mu)**2) + 2 * b)
#nu_s_hat = N + 2 * a


# 予測分布の確率密度を計算:式(3.76)
ln_C_St = gammaln(0.5 * (nu_s_hat + 1.0)) - gammaln(0.5 * nu_s_hat) # 正規化項(対数)
ln_term1 = 0.5 * np.log(lambda_s_hat / np.pi / nu_s_hat)
ln_term2 = - 0.5 * (nu_s_hat + 1.0) * np.log(1.0 + lambda_s_hat / nu_s_hat * (x_line - mu_s)**2)
predict_dens = np.exp(ln_C_St + ln_term1 + ln_term2)
#predict_dens = t.pdf(x=x_line, df=nu_s_hat, loc=mu_s, scale=np.sqrt(1 / lambda_s_hat))

#%%

# 予測分布を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, predict_dens, color='purple', label='predict') # 予測分布
plt.plot(x_line, model_dens, color='red', linestyle='--', label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle("Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + 
          ', \mu_s=' + str(mu_s) + 
          ', \hat{\lambda}_s=' + str(np.round(lambda_s_hat, 5)) + 
          ', \hat{\\nu}_s=' + str(nu_s_hat) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

### ・アニメーションによる推移の確認

# 3.3.2項で利用するライブラリ
import numpy as np
from scipy.stats import norm, gamma, t # 1次元ガウス分布, ガンマ分布, 1次元スチューデントのt分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# (既知の)平均パラメータを指定
mu = 25.0

# 真の精度パラメータを指定
lambda_truth = 0.01


# lambdaの事前分布のパラメータを指定
a = 1.0
b = 1.0


# 初期値による予測分布のパラメータを計算:式(3.79)
mu_st = mu
lambda_st = a / b
nu_st = 2.0 * a


# 作図用のlambdaの値を作成
lambda_line = np.linspace(0.0, 5.0 * lambda_truth, num=1000)

# 作図用のxの値を作成
x_line = np.linspace(
    mu - 4.0 * np.sqrt(1.0 / lambda_truth), 
    mu + 4.0 * np.sqrt(1.0 / lambda_truth), 
    num=1000
)

#%%

## 推論処理

# データ数(試行回数)を指定
N = 100

# 観測データの受け皿を作成
x_n = np.empty(N)

# 推移の記録用の受け皿を初期化
trace_a = [a]
trace_b = [b]
trace_posterior = [gamma.pdf(x=lambda_line, a=a, scale=1.0 / b)]
trace_lambda_st = [lambda_st]
trace_nu_st = [nu_st]
trace_predict = [t.pdf(x=x_line, df=nu_st, loc=mu_st, scale=np.sqrt(1.0 / lambda_st))]

# ベイズ推論
for n in range(N):
    # ガウス分布に従うデータを生成
    x_n[n] = np.random.normal(loc=mu, scale=np.sqrt(1.0 / lambda_truth), size=1)
    
    # lambdaの事前分布のパラメータを更新:式(3.69)
    a += 0.5
    b += 0.5 * (x_n[n] - mu)**2
    
    # lambdaの事前分布(ガンマ分布)を計算:式(2.56)
    trace_posterior.append(gamma.pdf(x=lambda_line, a=a, scale=1.0 / b))
    
    # 予測分布のパラメータを更新:式(3.79)
    mu_st = mu
    lambda_st = a / b
    nu_st = 2.0 * a
    
    # 予測分布(スチューデントのt分布)を計算:式(3.76)
    trace_predict.append(
        t.pdf(x=x_line, df=nu_st, loc=mu_st, scale=np.sqrt(1.0 / lambda_st))
    )
    
    # n回目の結果を記録
    trace_a.append(a)
    trace_b.append(b)
    trace_lambda_st.append(lambda_st)
    trace_nu_st.append(nu_st)

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
    
    # n回目のlambdaの事後分布を作図
    plt.plot(lambda_line, trace_posterior[n], color='purple', label='posterior') # lambdaの事後分布
    plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(trace_posterior), 
               color='red', linestyle='--', label='true val') # 真の値
    if n > 0: # 初回は除く
        plt.scatter(x=1.0 / x_n[:n-1]**2, y=np.repeat(0.0, n - 1)) # 観測データ
    plt.xlabel('$\lambda$')
    plt.ylabel('density')
    plt.suptitle('Gamma Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{a}=' + str(trace_a[n]) + 
              ', \hat{b}=' + str(np.round(trace_b[n], 1)) + '$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線
    plt.xlim((np.min(lambda_line), np.max(lambda_line))) # x軸の表示範囲

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_3_2_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度の確率密度を計算:式(2.64)
true_model = norm.pdf(x=x_line, loc=mu, scale=np.sqrt(1.0 / lambda_truth))

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の予測分布を作図
    plt.plot(x_line, trace_predict[n], label='predict', color='purple') # 予測分布
    plt.plot(x_line, true_model, color='red', linestyle='--', label='true model') # 真の分布
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[:n-1], y=np.repeat(0.0, n - 1), label='data') # 観測データ
    plt.xlabel('x')
    plt.ylabel('density')
    plt.suptitle("Student's t Distribution", fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \mu_s=' + str(mu_st) + 
              ', \hat{\lambda}_s=' + str(np.round(trace_lambda_st[n], 5)) + 
              ', \hat{\\nu}_s=' + str(trace_nu_st[n]) + '$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線
    plt.ylim(-0.001, 0.1) # y軸の表示範囲

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_3_2_Predict.gif")

#%%

print('end')

