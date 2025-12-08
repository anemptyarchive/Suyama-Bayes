
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.2
# 精度が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt


# %%

# ベイズ推論の実装 ---------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 既知のパラメータを指定
mu = 5.0

# 真のパラメータを指定
lambda_truth = 0.25
print(1.0/np.sqrt(lambda_truth))


# %%

#### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
x_size  = 1.0/np.sqrt(lambda_truth) # 基準値を指定
x_size *= 4.0 # 倍率を指定
#x_size  = max(x_size, (x_n-mu_truth).max()) # サンプルと比較
x_size  = np.ceil(x_size /u)*u # u単位で切り上げ
x_min   = mu - x_size
x_max   = mu + x_size
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# %%

#### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_vec = norm.pdf(x=x_vec, loc=mu, scale=1.0/np.sqrt(lambda_truth))


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
#lambda_max = 1.0
u = 0.5
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
x_n = np.random.normal(loc=mu, scale=1.0/np.sqrt(lambda_truth), size=N)


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

### 事後分布(ガンマ分布)の計算 -----

#### パラメータの計算 -----

# lambdaの事後分布のパラメータを計算:式(3.69)
a_hat = 0.5 * N + a
b_hat = 0.5 * np.sum((x_n - mu)**2) + b


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

### 予測分布(スチューデントのt分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.79')
mu_s         = mu
lambda_s_hat = a_hat / b_hat
nu_s_hat     = 2.0 * a_hat
#lambda_s_hat  = N + 2.0 * a
#lambda_s_hat /= np.sum((x_n - mu)**2) + 2.0 * b
#nu_s_hat      = N + 2.0 * a


# %%

#### 分布の計算 -----

# 予測分布の確率密度を計算
predict_dens_vec = t.pdf(x=x_vec, df=nu_s_hat, loc=mu_s, scale=1.0/np.sqrt(lambda_s_hat))


# %%

#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl  = f'$N = {N}, '
predict_param_lbl += f'\\mu = {mu:.3g}, \\lambda_{{truth}} = {lambda_truth:.3g}, '
predict_param_lbl += f'\\mu_s = {mu_s:.3g}, \\hat{{\\lambda}}_s = {lambda_s_hat:.3g}, \\hat{{\\nu}}_s = {nu_s_hat:.3g}$'

# 予測分布を作図
fig, ax = plt.subplots(figsize=(9, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle("Student's t Distribution", fontsize=20)

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


