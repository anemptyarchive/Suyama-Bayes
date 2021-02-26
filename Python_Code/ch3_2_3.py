# 3.2.3 ポアソン分布の学習と予測

#%%

# 3.2.3項で利用するライブラリ
import numpy as np
import math # 対数ガンマ関数:lgamma()
#from scipy.stats import poisson, gamma, nbinom # ポアソン分布,ガンマ分布,負の二項分布
import matplotlib.pyplot as plt

#%%

## 尤度(ポアソン分布)の設定

# 真のパラメータを指定
lambda_truth = 4.0

# 作図用のxの値を設定
x_line = np.arange(4 * int(lambda_truth))
print(x_line)

# 尤度を計算:式(2.37)
true_model = np.exp(
    x_line * np.log(lambda_truth) - [math.lgamma(x + 1) for x in x_line] - lambda_truth
)

# 尤度を計算:SciPy ver
#true_model = poisson.pmf(k=x_line, mu=lambda_truth)
print(np.round(true_model, 3))

#%%

# 尤度を作図
plt.bar(x=x_line, height=true_model, color='purple') # 尤度
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Distribution', fontsize=20)
plt.title('$\lambda=' + str(lambda_truth) + '$', loc='left')
plt.show()

#%%

##観測データの生成

# (観測)データ数を指定
N = 50

# ポアソン分布に従うデータをランダムに生成
x_n = np.random.poisson(lam=lambda_truth, size=N)
print(x_n)

#%%

# 観測データのヒストグラムを作図
plt.bar(x_line, [np.sum(x_n == x) for x in x_line]) # 観測データ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.show()

#%%

## 事前分布(ガンマ分布)の設定

# 事前分布のパラメータを指定
a = 1
b = 1

# 作図用のlambdaの値を設定
lambda_line = np.arange(2 * lambda_truth, 0.001)

# 事前分布を計算:式(2.56)
ln_C_gam = a * np.log(b) - math.lgamma(a) # 正規化項(対数)
prior = np.exp(ln_C_gam + (a - 1) * np.log(lambda_line) - b * lambda_line) # 確率密度

# 事前分布を計算:SciPy ver
#prior = gamma.pdf(x=lambda_line, a=a, scale=1 / b)

#%%

# 事前分布を作図
plt.plot(lambda_line, prior, color='purple') # 事前分布
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + "$", loc='left')
plt.show()

#%%

## 事後分布(ガンマ分布)の計算

# 事後分布のパラメータを計算:式(3.38)
a_hat = np.sum(x_n) + a
b_hat = N + b
print(a_hat)
print(b_hat)

# 事後分布を計算
ln_C_gam = a_hat * np.log(b_hat) - math.lgamma(a_hat) # 正規化項(対数)
posterior = np.exp(
    ln_C_gam + (a_hat - 1) * np.log(lambda_line) - b_hat * lambda_line
) # 確率密度

# 事後分布を計算:SciPy ver
#posterior = gamma.pdf(x=lambda_line, a=a_hat, scale=1 / b_hat)

#%%

# 事後分布を作図
plt.plot(lambda_line, posterior, color='purple') # 事後分布
plt.vlines(x=lambda_truth, ymin=0, ymax=max(posterior), color='red', linestyle='--') # 真のパラメータ
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(b_hat) + "$", loc='left')
plt.show()

#%%

## 予測分布(負の二項分布)の計算

# 予測分布のパラメータを計算:式(3.44')
r_hat = a_hat
p_hat = 1 / (b_hat + 1)
print(r_hat)
print(p_hat)

# 予測分布を計算:式(3.43)
ln_C_NB = np.array([math.lgamma(x + r_hat) - math.lgamma(x + 1) for x in x_line]) - math.lgamma(r_hat) # 正規化項(対数)
predict = np.exp(ln_C_NB + r_hat * np.log(1 - p_hat) + x_line * np.log(p_hat)) # 確率

# 予測分布を計算:SciPy ver
#predict = nbinom.pmf(k=x_line, n=r_hat, p=1 - p_hat) # 確率密度

#%%

# 予測分布を作図
plt.bar(x=x_line, height=true_model, label='true', alpha=0.5, color='white', 
        edgecolor='red', linestyle='--') # 真の分布
plt.bar(x=x_line, height=predict, label='predict', alpha=0.5, color='purple') # 予測分布
plt.xlabel('x')
plt.ylabel('prod')
plt.suptitle('Negative Binomial Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{r}=' + str(r_hat) + ', \hat{p}=' + str(np.round(p_hat, 3)) + '$', loc='left')
plt.show()


#%%

## アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import poisson, gamma, nbinom # ポアソン分布,ガンマ分布,負の二項分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## ベイズ推論

# 真のパラメータを指定
lambda_truth = 4.0

# 事前分布のパラメータを指定
a = 1
b = 1

# 初期値による予測分布のパラメータを計算:式(3.44)
r = a
p = 1 / (b + 1)

# データ数(試行回数)を指定
N = 100

# 作図用の値を設定
lambda_line = np.arange(0, 2 * lambda_truth, 0.001)
x_line = np.arange(4 * int(lambda_truth))

# 推移の記録用の受け皿を初期化
x_n = np.empty(N)
trace_a = [a]
trace_b = [b]
trace_posterior = [gamma.pdf(x=lambda_line, a=a, scale=1 / b)]
trace_r = [r]
trace_p = [p]
trace_predict = [nbinom.pmf(k=x_line, n=r, p=1 - p)]

# ベイズ推論
for n in range(N):
    # ポアソン分布に従うデータを生成
    x_n[n] = np.random.poisson(lam=lambda_truth, size=1)
    
    # 事後分布のパラメータを計算:式(3.38)
    a += x_n[n]
    b += 1
    
    # 事後分布(ガンマ分布)を計算:式(2.56)
    trace_posterior.append(gamma.pdf(x=lambda_line, a=a, scale=1 / b)) # 確率密度
    
    # 予測分布のパラメータを計算:式(3.44)
    r = a
    p = 1 / (b + 1)
    
    # 予測分布(負の二項分布)を計算:式(3.43)
    trace_predict.append(nbinom.pmf(k=x_line, n=r, p=1 - p)) # 確率
    
    # 超パラメータを記録
    trace_a.append(a)
    trace_b.append(b)
    trace_r.append(r)
    trace_p.append(p)

# 観測データを確認
print(x_n)

#%%

## 事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の事後分布を作図
    plt.plot(lambda_line, trace_posterior[n], color='purple') # 事後分布
    plt.vlines(x=lambda_truth, ymin=0, ymax=np.nanmax(trace_posterior), 
               color='red', linestyles='--') # 真のパラメータ
    plt.xlabel('$\lambda$')
    plt.ylabel('density')
    plt.suptitle('Gamma Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{a}=' + str(trace_a[n]) + ', \hat{b}=' + str(trace_b[n]) + "$", loc='left')

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
    
    # nフレーム目の予測分布を作図
    plt.bar(x=x_line, height=true_model, label='true', alpha=0.5, 
            color='white', edgecolor='red', linestyle='--') # 真の分布
    plt.bar(x=x_line, height=trace_predict[n], label='predict', alpha=0.5, 
            color='purple') # 予測分布
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.suptitle('Bernoulli Distribution', fontsize=20)
    plt.suptitle('Negative Binomial Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{r}=' + str(trace_r[n]) + ', \hat{p}=' + str(np.round(trace_p[n], 3)) + '$', loc='left')
    plt.ylim(0.0, np.nanmax(trace_predict)) # y軸の表示範囲
    plt.legend()

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_2_3_Predict.gif")

#%%

print('end')


