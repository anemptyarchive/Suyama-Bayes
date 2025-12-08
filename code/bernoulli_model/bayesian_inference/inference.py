
# ベルヌーイモデル -------------------------------------------------------------

# chapter 3.2.1
# ベイズ推論
# 推論アルゴリズムの実装


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# %%

# ベイズ推論の実装 -------------------------------------------------------------

### 生成分布(ベルヌーイ分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
mu_truth = 0.25


# %%

#### 変数の設定 -----

# x軸の範囲を設定
x_min = 0 # (固定)
x_max = 1 # (固定)

# x軸の値を作成
x_vec = np.arange(start=x_min, stop=x_max+1, step=1)


# %%

#### 分布の計算 -----

# 生成分布の確率を計算
model_prob_vec = np.array([1.0-mu_truth, mu_truth])


# %%

### 事前分布(ベータ分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
a = 1.0
b = 1.0


# %%

#### 変数の設定 -----

# μ軸の範囲を設定
mu_min = 0 # (固定)
mu_max = 1 # (固定)

# μ軸の値を作成
mu_vec = np.linspace(start=mu_min, stop=mu_max, num=1001)


# %%

#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_dens_vec = beta.pdf(x=mu_vec, a=a, b=b)


# %%

### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N = 50

# 観測データを生成
x_n = np.random.binomial(n=1, p=mu_truth, size=N)


# %%

### データの集計 -----

# 相対度数を集計
obs_relfreq_vec = np.array([N-np.sum(x_n), np.sum(x_n)]) / N


# %%

### 事後分布(ベータ分布)の計算 -----

#### パラメータの計算 -----

# 事後分布のパラメータを計算:式(3.15)
a_hat = np.sum(x_n) + a
b_hat = N - np.sum(x_n) + b


# %%

#### 分布の計算 -----

# 事後分布の確率密度を計算
posterior_dens_vec = beta.pdf(x=mu_vec, a=a_hat, b=b_hat)


# %%

#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl  = f'$N = {N}, '
posterior_param_lbl += f'\\mu_{{truth}} = {mu_truth:.3g}, '
posterior_param_lbl += f'a = {a:.3g}, b = {b:.3g}, '
posterior_param_lbl += f'\\hat{{a}} = {a_hat:.3g}, \\hat{{b}} = {b_hat:.3g}$'

# 事後分布を作図
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Beta distribution', fontsize=20)

ax.axvline(
    x=mu_truth, 
    color='red', linewidth=1.0, linestyle='--', 
    label='true parameter', zorder=10
) # 真のパラメータ
ax.plot(
    mu_vec, prior_dens_vec, 
    color='purple', linewidth=1.0, linestyle=':', 
    label='prior distribution', zorder=11
) # 事前分布
ax.plot(
    mu_vec, posterior_dens_vec, 
    color='purple', linewidth=1.0, 
    label='posterior distribution', zorder=12
) # 事後分布
ax.set_xlabel('$\mu$')
ax.set_ylabel('density')
ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
ax.legend(prop={'size': 8})
ax.grid(zorder=0)
ax.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)

ax2 = ax.twiny() # 第2軸の設定用
ax2.set_xticks(ticks=[mu_truth], labels=['$\mu_{truth}$']) # パラメータラベル
ax2.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)

plt.show()


# %%

### 予測分布(ベルヌーイ分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.19')
mu_s_hat = a_hat / (a_hat + b_hat)
#mu_s_hat = (np.sum(x_n) + a) / (N + a + b)


# %%

#### 分布の計算 -----

# 予測分布の確率を計算
predict_prob_vec = np.array([1.0-mu_s_hat, mu_s_hat])


# %%

#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl  = f'$N = {N}, '
predict_param_lbl += f'\\mu_{{truth}} = {mu_truth:.3g}, '
predict_param_lbl += f'\\hat{{\\mu}}_{{*}} = {mu_s_hat:.3g}$'

# 予測分布を作図
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Bernoulli distribution', fontsize=20)

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


