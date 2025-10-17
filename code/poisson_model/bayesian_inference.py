# 3.2.3 ポアソン分布の学習と予測

#%%

# 3.2.3項で利用するライブラリ
import numpy as np
from scipy.special import gammaln # 対数ガンマ関数
#from scipy.stats import poisson, gamma, nbinom # ポアソン分布, ガンマ分布, 負の二項分布
import matplotlib.pyplot as plt

#%%

## 尤度(ポアソン分布)の設定

# 真のパラメータを指定
lambda_truth = 4.0

# 作図用のxの値を作成
x_line = np.arange(4.0 * lambda_truth)

# 尤度の確率を計算:式(2.37)
model_prob = np.exp(
    x_line * np.log(lambda_truth) - gammaln(x_line + 1) - lambda_truth
)
#model_prob = poisson.pmf(k=x_line, mu=lambda_truth)

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Distribution', fontsize=20)
plt.title('$\lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# ポアソン分布に従うデータをランダムに生成
x_n = np.random.poisson(lam=lambda_truth, size=N)

#%%

# 度数をカウント
x_uni, x_cnt = np.unique(x_n, return_counts=True)

# 観測データのヒストグラムを作成
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, 
        color='white', edgecolor='red', linestyle='--', label='true model') # 真の分布
plt.bar(x=x_uni, height=x_cnt/N, 
        alpha=0.9, label='observation data') # 観測データ:(相対度数)
#plt.bar(x=x_uni, height=x_cnt, label='observation data') # 観測データ:(度数)
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Poisson Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事前分布(ガンマ分布)の設定

# 事前分布のパラメータを指定
a = 1.0
b = 1.0


# 作図用のlambdaの値を作成
lambda_line = np.arange(0.0, 2.0 * lambda_truth, step=0.001)

# 事前分布の確率密度を計算:式(2.56)
ln_C_Gam = a * np.log(b) - gammaln(a) # 正規化項(対数)
prior_dens = np.exp(ln_C_Gam + (a - 1) * np.log(lambda_line) - b * lambda_line)
#prior_dens = gamma.pdf(x=lambda_line, a=a, scale=1 / b)

#%%

# 事前分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, prior_dens, color='purple', label='prior') # 事前分布
plt.vlines(x=lambda_truth, ymin=0, ymax=np.nanmax(prior_dens), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + "$", loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事後分布(ガンマ分布)の計算

# 事後分布のパラメータを計算:式(3.38)
a_hat = np.sum(x_n) + a
b_hat = N + b


# 事後分布の確率密度を計算:式(2.56)
ln_C_Gam = a_hat * np.log(b_hat) - gammaln(a_hat) # 正規化項(対数)
posterior_dens = np.exp(
    ln_C_Gam + (a_hat - 1) * np.log(lambda_line) - b_hat * lambda_line
)
#posterior = gamma.pdf(x=lambda_line, a=a_hat, scale=1 / b_hat)

#%%

# 事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, posterior_dens, color='purple', label='posterior') # 事後分布
plt.vlines(x=lambda_truth, ymin=0, ymax=np.nanmax(posterior_dens), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(b_hat) + "$", loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 予測分布(負の二項分布)の計算

# 予測分布のパラメータを計算:式(3.44')
r_hat = a_hat
p_hat = 1 / (b_hat + 1)


# 予測分布の確率を計算:式(3.43)
ln_C_NB = gammaln(x_line + r_hat) - gammaln(x_line + 1) - gammaln(r_hat) # 正規化項(対数)
predict_prob = np.exp(ln_C_NB + r_hat * np.log(1 - p_hat) + x_line * np.log(p_hat)) # 確率
#predict_prob = nbinom.pmf(k=x_line, n=r_hat, p=1 - p_hat) # 確率密度

#%%

# 予測分布を作図
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, 
        color='white', edgecolor='red', linestyle='--', label='true model') # 真の分布
plt.bar(x=x_line, height=predict_prob, 
        alpha=0.6, color='purple', label='predict') # 予測分布
plt.xlabel('x')
plt.ylabel('prod')
plt.suptitle('Negative Binomial Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{r}=' + str(r_hat) + 
          ', \hat{p}=' + str(np.round(p_hat, 3)) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()


#%%

### アニメーションによる推移の確認

# 3.2.3項で利用するライブラリ
import numpy as np
from scipy.stats import poisson, gamma, nbinom # ポアソン分布, ガンマ分布, 負の二項分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# 真のパラメータを指定
lambda_truth = 4.0

# 事前分布のパラメータを指定
a = 1.0
b = 1.0

# 初期値による予測分布のパラメータを計算:式(3.44)
r = a
p = 1.0 / (b + 1.0)


# 作図用のlambdaの値を作成
lambda_line = np.linspace(0.0, 2.0 * lambda_truth, num=1000)

# 作図用のxの値を作成
x_line = np.arange(4.0 * lambda_truth)

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
trace_r = [r]
trace_p = [p]
trace_predict = [nbinom.pmf(k=x_line, n=r, p=1.0 - p)]

# ベイズ推論
for n in range(N):
    # ポアソン分布に従うデータを生成
    x_n[n] = np.random.poisson(lam=lambda_truth, size=1)
    
    # 事後分布のパラメータを更新:式(3.38)
    a += x_n[n]
    b += 1.0
    
    # 事後分布(ガンマ分布)を計算:式(2.56)
    trace_posterior.append(gamma.pdf(x=lambda_line, a=a, scale=1 / b)) # 確率密度
    
    # 予測分布のパラメータを更新:式(3.44)
    r = a
    p = 1.0 / (b + 1.0)
    
    # 予測分布(負の二項分布)を計算:式(3.43)
    trace_predict.append(nbinom.pmf(k=x_line, n=r, p=1.0 - p)) # 確率
    
    # 超パラメータを記録
    trace_a.append(a)
    trace_b.append(b)
    trace_r.append(r)
    trace_p.append(p)

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
    
    # n回目の事後分布を作図
    plt.plot(lambda_line, trace_posterior[n], 
             color='purple', label='posterior') # 事後分布
    plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(trace_posterior), 
               color='red', linestyles='--', label='true val') # 真のパラメータ
    if n > 0: # 初回は除く
        plt.scatter(x_n[n-1], y=0.0, s=100, label='data') # 観測データ
    plt.xlabel('$\lambda$')
    plt.ylabel('density')
    plt.suptitle('Gamma Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{a}=' + str(trace_a[n]) + 
              ', \hat{b}=' + str(trace_b[n]) + "$", loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_2_3_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 尤度を計算:式(2.37)
true_model = poisson.pmf(k=x_line, mu=lambda_truth)

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の予測分布を作図
    plt.bar(x=x_line, height=true_model, 
            color='white', edgecolor='red', linestyle='--', label='true model', zorder=0) # 真の分布
    plt.bar(x=x_line, height=trace_predict[n], 
            alpha=0.6, color='purple', label='predict', zorder=1) # 予測分布
    if n > 0: # 初回は除く
        plt.scatter(x=x_n[n-1], y=0.0, s=100, label='data', zorder=2) # 観測データ
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.suptitle('Bernoulli Distribution', fontsize=20)
    plt.suptitle('Negative Binomial Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{r}=' + str(trace_r[n]) + 
              ', \hat{p}=' + str(np.round(trace_p[n], 5)) + '$', loc='left')
    plt.ylim(-0.01, np.nanmax(trace_predict)) # y軸の表示範囲
    plt.legend() # 凡例

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_2_3_Predict.gif")

#%%

print('end')

