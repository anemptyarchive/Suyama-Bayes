
# ポアソンモデル ----------------------------------------------------------------

# chapter 3.2.3
# ベイズ推論
# 推論アルゴリズムの実装


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import poisson, gamma, nbinom
import matplotlib.pyplot as plt


# %%

# ベイズ推論の実装 ---------------------------------------------------------------

### 生成分布(ポアソン分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
lambda_truth = 4.0


# %%

#### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
x_min = 0.0
x_max = lambda_truth # 基準値を指定
x_max *= 3.0 # 倍率を指定
#x_max = max(x_max, x_n.max()) # サンプルと比較
x_max = np.ceil(x_max /u)*u # u単位で切り上げ
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.arange(start=x_min, stop=x_max+1, step=1)


# %%

#### 分布の計算 -----

# 生成分布の確率を計算
model_prob_vec = poisson.pmf(k=x_vec, mu=lambda_truth)


# %%

### 事前分布(ガンマ分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
a = 1.0
b = 1.0


# %%

#### 変数の設定 -----

# λ軸の範囲を設定
lambda_min = 0.0
lambda_max = lambda_truth # 基準値を指定
lambda_max *= 3.0 # 倍率を指定
lambda_max = np.ceil(lambda_max /u)*u # u単位で切り上げ
print('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec = np.linspace(start=lambda_min, stop=lambda_max, num=1001)


# %%

#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_dens_vec = gamma.pdf(x=lambda_vec, a=a, scale=1.0/b)


# %%

### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N = 50

# 観測データを生成
x_n = np.random.poisson(lam=lambda_truth, size=N)


# %%

### データの集計 -----

# 相対度数を集計
obs_relfreq_vec = np.array([np.sum(x_n == x) for x in x_vec]) / N


# %%

### 事後分布(ガンマ分布)の計算 -----

#### パラメータの計算 -----

# 事後分布のパラメータを計算:式(3.38)
a_hat = np.sum(x_n) + a
b_hat = N + b


# %%

#### 分布の計算 -----

# 事後分布の確率密度を計算
posterior_dens_vec = gamma.pdf(x=lambda_vec, a=a_hat, scale=1.0/b_hat)


# %%

#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl  = f'$N = {N}, '
posterior_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.3g}, '
posterior_param_lbl += f'a = {a:.3g}, b = {b:.3g}, '
posterior_param_lbl += f'\\hat{{a}} = {a_hat:.3g}, \\hat{{b}} = {b_hat:.3g}$'

# 事後分布を作図
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Gamma distribution', fontsize=20)

ax.axvline(
    x=lambda_truth, 
    color='red', linewidth=1.0, linestyle='--', 
    label='true parameter', zorder=10
) # 真のパラメータ
ax.plot(
    lambda_vec, prior_dens_vec, 
    color='purple', linewidth=1.0, linestyle=':', 
    label='prior distribution', zorder=11
) # 事前分布
ax.plot(
    lambda_vec, posterior_dens_vec, 
    color='purple', linewidth=1.0, 
    label='posterior distribution', zorder=12
) # 事後分布
ax.set_xlabel('$\lambda$')
ax.set_ylabel('density')
ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
ax.legend(prop={'size': 8})
ax.grid(zorder=0)
ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)

ax2 = ax.twiny() # 第2軸の設定用
ax2.set_xticks(ticks=[lambda_truth], labels=['$\lambda_{truth}$']) # パラメータラベル
ax2.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)

plt.show()


# %%

### 予測分布(負の二項分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.44')
r_hat = a_hat
q_hat = 1.0 / (1.0 + b_hat)
p_hat = b_hat / (1.0 + b_hat)
#r_hat = np.sum(x_n) + a
#q_hat = 1.0 / (1.0 + N + b)
#p_hat = (N + b) / (1.0 + N + b)
#p_hat = 1 - q_hat


# %%

#### 分布の計算 -----

# 予測分布の確率を計算
predict_prob_vec = nbinom.pmf(k=x_vec, n=r_hat, p=p_hat)


# %%

#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl  = f'$N = {N}, '
predict_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.3g}, '
predict_param_lbl += f'\\hat{{r}} = {r_hat:.3g}, \\hat{{p}} = {p_hat:.3g}$'

# 予測分布を作図
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Negative Binomialson distribution', fontsize=20)

ax.bar(
    x=x_vec, height=model_prob_vec, 
    facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--', 
    label='true model', zorder=10
) # 真の分布
ax.bar(
    x=x_vec, height=predict_prob_vec, 
    color='purple', alpha=0.5, 
    label='predict', zorder=11
) # 予測分布
ax.set_xticks(ticks=x_vec) # x軸目盛
ax.set_xlabel('$x$')
ax.set_ylabel('probability')
ax.set_title(predict_param_lbl, loc='left') # パラメータラベル
ax.legend(prop={'size': 8})
ax.grid(zorder=0)

plt.show()


# %%


