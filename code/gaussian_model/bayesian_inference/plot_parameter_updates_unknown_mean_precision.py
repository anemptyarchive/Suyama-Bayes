
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.3
# 平均と精度が未知の場合
# ベイズ推論
# 学習推移の可視化


# %% 

### ・アニメーションによる推移の確認

# 3.3.3項で利用するライブラリ
import numpy as np
from scipy.stats import norm, gamma, t # 1次元ガウス分布, ガンマ分布, 1次元スチューデントのt分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# 真の平均パラメータを指定
mu_truth = 25.0

# 真の精度パラメータを指定
lambda_truth = 0.01


# muの事前分布のパラメータを指定
m = 0.0
beta = 1.0

# lambdaの事前分布のパラメータを指定
a = 1.0
b = 1.0


# 初期値による予測分布のパラメータを計算:式(3.95)
mu_st = m
lambda_st = beta * a / (1.0 + beta) / b
nu_st = 2.0 * a


# 作図用のmuの値を設定
mu_line = np.linspace(mu_truth - 50, mu_truth + 50, num=1000)

# 作図用のlambdaの値を設定
lambda_line = np.linspace(0.0, 5.0 * lambda_truth, num=1000)

# 作図用のxの値を作成
x_line = np.linspace(
    mu_truth - 4.0 * np.sqrt(1.0 / lambda_truth), 
    mu_truth + 4.0 * np.sqrt(1.0 / lambda_truth), 
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
trace_beta = [beta]
trace_posterior_mu = [norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1.0 / (beta * a / b)))]
trace_a = [a]
trace_b = [b]
trace_posterior_lambda = [gamma.pdf(x=lambda_line, a=a, scale=1.0 / b)]
trace_mu_st = [mu_st]
trace_lambda_st = [lambda_st]
trace_nu_st = [nu_st]
trace_predict = [t.pdf(x=x_line, df=nu_st, loc=mu_st, scale=np.sqrt(1.0 / lambda_st))]

# ベイズ推論
for n in range(N):
    # ガウス分布に従うデータを生成
    x_n[n] = np.random.normal(loc=mu_truth, scale=np.sqrt(1.0 / lambda_truth), size=1)
    
    # muの事後分布のパラメータを更新:式(3.83)
    old_beta = beta
    old_m = m
    beta += 1.0
    m = (x_n[n] + old_beta * m) / beta
    
    # lambdaの事後分布のパラメータを更新:式(3.88)
    a += 0.5
    b += 0.5 * (x_n[n]**2 + old_beta * old_m**2 - beta * m**2)
    
    # lambdaの期待値を計算:式(2.59)
    E_lambda = a / b
    
    # muの事前分布分布(ガウス分布)を計算:式(2.64)
    trace_posterior_mu.append(
        norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1.0 / (beta * a / b)))
    )
    
    # lambdaの事前分布(ガンマ)を計算:式(2.56)
    trace_posterior_lambda.append(
        gamma.pdf(x=lambda_line, a=a, scale=1.0 / b)
    )
    
    # 予測分布のパラメータを更新:式(3.95)
    mu_st = m
    lambda_st = beta * a / (1.0 + beta) / b
    nu_st = 2.0 * a
    
    # 予測分布(スチューデントのt分布)を計算:式(3.76)
    trace_predict.append(
        t.pdf(x=x_line, df=nu_st, loc=mu_st, scale=np.sqrt(1.0 / lambda_st))
    )
    
    # n回目の結果を記録
    trace_m.append(m)
    trace_beta.append(beta)
    trace_a.append(a)
    trace_b.append(b)
    trace_mu_st.append(mu_st)
    trace_lambda_st.append(lambda_st)
    trace_nu_st.append(nu_st)

#%%

## muの事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior_mu(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目のmuの事後分布を作図
    plt.plot(mu_line, trace_posterior_mu[n], color='purple', label='$\mu$ posterior') # muの事後分布
    plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(trace_posterior_mu), 
               color='red', linestyle='--', label='true val') # 真の値
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[:n-1], y=np.repeat(0.0, n - 1), label='data') # 観測データ
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.suptitle('Gaussian Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{m}=' + str(np.round(trace_m[n], 1)) + 
              ', \hat{\\beta}=' + str(trace_beta[n]) + 
              ', E[\lambda]=' + str(np.round(trace_a[n] / trace_b[n], 5)) + '$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線

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
    
    # n回目のlambdaの事後分布を作図
    plt.plot(lambda_line, trace_posterior_lambda[n], 
             color='purple', label='$\lambda$ posterior') # lambdaの事後分布
    plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(trace_posterior_lambda), 
               color='red', linestyle='--', label='true val') # 真の値
    if n > 0: # 初回は除く
        plt.scatter(x=1.0 / x_n[:n-1]**2, y=np.repeat(0.0, n - 1), label='data') # 観測データ
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
posterior_anime = animation.FuncAnimation(fig, update_posterior_lambda, frames=N + 1, interval=100)
posterior_anime.save("ch3_3_3_Posterior_lambda.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度の確率密度を計算:式(2.64)
true_model = norm.pdf(x=x_line, loc=mu_truth, scale=np.sqrt(1.0 / lambda_truth))

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の予測分布を作図
    plt.plot(x_line, trace_predict[n], color='purple', label='predict') # 予測分布
    plt.plot(x_line, true_model, color='red', linestyle='--', label='true model') # 真の分布
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[:n-1], y=np.repeat(0.0, n - 1), label='data') # 観測データ
    plt.xlabel('x')
    plt.ylabel('density')
    plt.suptitle("Student's t Distribution", fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\mu}_s=' + str(np.round(trace_mu_st[n], 1)) + 
              ', \hat{\lambda}_s=' + str(np.round(trace_lambda_st[n], 5)) + 
              ', \hat{\\nu}_s=' + str(trace_nu_st[n]) + '$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線
    plt.ylim(-0.001, 0.1) # y軸の表示範囲

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_3_3_Predict.gif")

#%%

print('end')

