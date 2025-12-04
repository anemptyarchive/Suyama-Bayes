
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.1
# 平均が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt


# %%

# ベイズ推論の実装 ---------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
mu_truth = 25.0

# 既知のパラメータを指定
lmd = 0.01
print(1.0/np.sqrt(lmd))


# %%

#### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
x_size  = 1.0/np.sqrt(lmd) # 基準値を指定
x_size *= 4.0 # 倍率を指定
#x_size  = max(x_size, (x_n-mu_truth).max()) # サンプルと比較
x_size  = np.ceil(x_size /u)*u # u単位で切り上げ
x_min   = mu_truth - x_size
x_max   = mu_truth + x_size
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# %%

#### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_vec = norm.pdf(x=x_vec, loc=mu_truth, scale=1.0/np.sqrt(lmd))


# %%

### 事前分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 事前分布のパラメータを指定
m         = 0.0
lambda_mu = 0.001


# %%

#### 変数の設定 -----

# μ軸の範囲を設定
mu_min = x_min
mu_max = x_max
print('μ size:', mu_min, mu_max)

# μ軸の値を作成
mu_vec = np.linspace(start=mu_min, stop=mu_max, num=1001)


# %%

#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_dens_vec = norm.pdf(x=mu_vec, loc=m, scale=1.0/np.sqrt(lambda_mu))


# %%

### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N = 50

# 観測データを生成
x_n = np.random.normal(loc=mu_truth, scale=1.0/np.sqrt(lmd), size=N)


# %%

### データの集計 -----

# 階級数を指定
bin_num = 40

# 階級幅を計算
bin_size = (x_max - x_min) / bin_num

# 境界値の範囲を設定
bin_min = x_min - 0.5*bin_size
bin_max = x_max + 0.5*bin_size

# 密度を集計
obs_dens_vec, bin_vec = np.histogram(a=x_n, bins=bin_num+1, range=(bin_min, bin_max), density=True)

# 階級値を作成
center_vec = bin_vec[:-1] + 0.5*bin_size


# %%

### 事後分布(ガウス分布)の計算 -----

#### パラメータの計算 -----

# 事後分布のパラメータを計算:式(3.53, 3.54)
lambda_mu_hat = N * lmd + lambda_mu
m_hat         = (lmd * np.sum(x_n) + m * lambda_mu) / lambda_mu_hat


# %%

#### 分布の計算 -----

# 事後分布の確率密度を計算
posterior_dens_vec = norm.pdf(x=mu_vec, loc=m_hat, scale=1.0/np.sqrt(lambda_mu_hat))


# %%

#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl  = f'$N = {N}, '
posterior_param_lbl += f'\\mu_{{truth}} = {mu_truth:.3g}, '
posterior_param_lbl += f'm = {m:.3g}, \\lambda_{{\\mu}} = {lambda_mu:.3g}, '
posterior_param_lbl += f'\\hat{{m}} = {m_hat:.3g}, \\hat{{\\lambda}}_{{\\mu}} = {lambda_mu_hat:.3g}$'

# 事後分布を作図
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Gaussian distribution', fontsize=20)

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

### 予測分布(ガウス分布)を計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.62')
mu_s_hat     = m_hat
lambda_s_hat = lmd * lambda_mu_hat / (lmd + lambda_mu_hat)
#mu_s_hat      = lmd * np.sum(x_n) + m * lambda_mu
#mu_s_hat     /= N * lmd + lambda_mu
#lambda_s_hat  = (N * lmd + lambda_mu) * lmd
#lambda_s_hat /= (N + 1) * lmd + lambda_mu


# %%

#### 分布の計算 -----

# 予測分布の確率密度を計算
predict_dens_vec = norm.pdf(x=x_vec, loc=mu_s_hat, scale=1.0/np.sqrt(lambda_s_hat))


# %%

#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl  = f'$N = {N}, '
predict_param_lbl += f'\\mu_{{truth}} = {mu_truth:.3g}, \\lambda = {lmd:.3g}, '
predict_param_lbl += f'\\hat{{\\mu}}_{{*}} = {mu_s_hat:.3g}, \\hat{{\\lambda}}_{{*}} = {lambda_s_hat:.3g}$'

# 予測分布を作図
fig, ax = plt.subplots(figsize=(9, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Gaussian distribution', fontsize=20)

ax.plot(
    x_vec, model_dens_vec, 
    color='red', linewidth=1.0, linestyle='--', 
    label='true model', zorder=10
) # 真の分布
ax.plot(
    x_vec, predict_dens_vec, 
    color='purple', linewidth=1.0, 
    label='predict distribution', zorder=11
) # 予測分布
ax.set_xlabel('$x$')
ax.set_ylabel('density')
ax.set_title(predict_param_lbl, loc='left') # パラメータラベル
ax.legend(prop={'size': 8})
ax.grid(zorder=0)

plt.show()


# %%


