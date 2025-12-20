
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.3
# 平均と精度が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# %%

# ライブラリを読込
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# %%

# ベイズ推論の実装 ---------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
mu_truth = 5.0
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
x_min   = mu_truth - x_size
x_max   = mu_truth + x_size
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# %%

#### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_vec = norm.pdf(x=x_vec, loc=mu_truth, scale=1.0/np.sqrt(lambda_truth))


# %%

### 事前分布(ガンマ分布)の設定 -----

#### パラメータの設定 -----

# μの事前分布のパラメータを指定
m    = 0.0
beta = 1.0

# λの事前分布のパラメータを指定
a = 1.0
b = 1.0


#### 変数の設定 -----

# μ軸の範囲を設定
mu_min = x_min
mu_max = x_max
print('μ size:', mu_min, mu_max)

# μ軸の値を作成
mu_vec = np.linspace(start=mu_min, stop=mu_max, num=201)


# λ軸の範囲を設定
lambda_min = 0.0
#lambda_max = 1.0
u = 0.5
lambda_max = lambda_truth # 基準値を指定
lambda_max *= 3.0 # 倍率を指定
lambda_max = np.ceil(lambda_max /u)*u # u単位で切り上げ
print('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec = np.linspace(start=lambda_min, stop=lambda_max, num=201)


# 格子点を作成
mu_mat, lambda_mat = np.meshgrid(mu_vec, lambda_vec)


# %%

### 分布の計算 -----

# 事前分布の確率密度を計算
N_dens_mat     = norm.pdf(x=mu_mat, loc=m, scale=1.0/np.sqrt(beta*lambda_mat))
Gam_dens_mat   = gamma.pdf(x=lambda_mat, a=a, scale=1.0/b)
prior_dens_mat = N_dens_mat * Gam_dens_mat


# %%

### 観測データの生成 -----

#### データの生成 -----

# データ数を指定
N = 50

# 観測データを生成
x_n = np.random.normal(loc=mu_truth, scale=1.0/np.sqrt(lambda_truth), size=N)


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

### 事後分布(ガウス・ガンマ分布)の計算 -----

#### パラメータの計算 -----

# μの事後分布のパラメータを計算:式(3.83)
beta_hat = N + beta
m_hat    = (np.sum(x_n) + beta * m) / beta_hat

# λの事後分布のパラメータを計算:式(3.88)
a_hat = 0.5 * N + a
b_hat = 0.5 * (np.sum(x_n**2) + beta * m**2 - beta_hat * m_hat**2) + b


# %%

### 分布の計算  -----

# 事後分布の確率密度を計算
N_dens_mat         = norm.pdf(x=mu_mat, loc=m_hat, scale=1.0/np.sqrt(beta_hat*lambda_mat))
Gam_dens_mat       = gamma.pdf(x=lambda_mat, a=a_hat, scale=1.0/b_hat)
posterior_dens_mat = N_dens_mat * Gam_dens_mat


# %%

#### 分布の作図 -----

# 確率密度軸の範囲を設定
u = 0.5
dens_max = max(prior_dens_mat.max(), posterior_dens_mat.max())
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)


# 事後分布のラベルを作成
posterior_param_lbl  = f'$N = {N}, '
posterior_param_lbl += f'\\mu_{{truth}} = {mu_truth:.3g}, \\lambda_{{truth}} = {lambda_truth:.3g}$\n'
posterior_param_lbl += f'$m = {m:.3g}, \\beta = {beta:.3g}, a = {a:.3g}, b = {b:.3g}$\n'
posterior_param_lbl += f'$\\hat{{m}} = {m_hat:.3g}, \\hat{{\\beta}} = {beta_hat:.3g}, \\hat{{a}} = {a_hat:.3g}, \\hat{{b}} = {b_hat:.3g}$'


# %%

## 等高線図

# 事後分布を作図
fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Gaussian-Gamma distribution', fontsize=20)

ax.plot(
    0.0, 0.0, 
    color=cm.viridis(X=0.0), linewidth=1.0, linestyle=':', 
    label='prior distribution', zorder=10
) # (凡例表示用のダミー)
ax.plot(
    0.0, 0.0, 
    color=cm.viridis(X=0.0), linewidth=1.0, linestyle='-', 
    label='posterior distribution', zorder=10
) # (凡例表示用のダミー)
prior_cs = ax.contour(
    mu_mat, lambda_mat, prior_dens_mat, 
    cmap='viridis', vmin=0.0, vmax=dens_max, 
    linewidths=1.0, linestyles=':', 
    zorder=11
) # 事前分布
posterior_cs = ax.contourf(
    mu_mat, lambda_mat, posterior_dens_mat, 
    cmap='viridis', vmin=0.0, vmax=dens_max, 
    alpha=0.5, 
    zorder=12
) # 事後分布
ax.axvline(
    x=mu_truth, 
    color='red', linewidth=1.0, linestyle='--', 
    label='true parameter', zorder=13
) # 真のパラメータ
ax.axhline(
    y=lambda_truth, 
    color='red', linewidth=1.0, linestyle='--', 
    zorder=13
) # 真のパラメータ
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
fig.colorbar(posterior_cs, ax=ax, shrink=1.0, label='density')
fig.colorbar(prior_cs, ax=ax, shrink=1.0, label='density')
ax.legend(prop={'size': 8})
ax.grid(zorder=0)
ax.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)

ax2x = ax.twiny() # 第2軸の設定用
ax2x.set_xticks(ticks=[mu_truth], labels=['$\mu_{truth}$']) # パラメータラベル
ax2x.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)

ax2y = ax.twinx() # 第2軸の設定用
ax2y.set_yticks(ticks=[lambda_truth], labels=['$\lambda_{truth}$']) # パラメータラベル
ax2y.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)

plt.show()


# %%

## 曲面図

# 事後分布・事後分布を作図
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(9, 6), dpi=100, facecolor='white', 
    constrained_layout=True, subplot_kw={'projection': '3d'}
)
fig.suptitle('Gaussian-Gamma distribution', fontsize=20)

# 事前分布を描画
ax = axes[0]
ax.plot(
    [mu_truth, mu_truth], [lambda_min, lambda_max], [0.0, 0.0], 
    color='red', linewidth=1.0, linestyle='--', 
    label='true parameter', zorder=10
) # 真のパラメータ
ax.plot(
    [mu_min, mu_max], [lambda_truth, lambda_truth], [0.0, 0.0], 
    color='red', linewidth=1.0, linestyle='--', 
    zorder=10
) # 真のパラメータ
ax.contour(
    X=mu_mat, Y=lambda_mat, Z=prior_dens_mat, offset=0.0, 
    cmap='viridis', vmin=0.0, vmax=dens_max, 
    linewidths=0.8, linestyles=':', 
    zorder=11
) # 事前分布
ax.plot_surface(
    X=mu_mat, Y=lambda_mat, Z=prior_dens_mat, 
    cmap='viridis', vmin=0.0, vmax=dens_max, 
    alpha=0.5, 
    label='prior distribution', zorder=12
) # 事前分布
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('density')
ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
ax.legend(prop={'size': 8})
ax.grid(zorder=0)
ax.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)
ax.set_zlim(zmin=0.0, zmax=dens_max)          # (目盛の共通化用)

# 事後分布を描画
ax = axes[1]
ax.plot(
    [mu_truth, mu_truth], [lambda_min, lambda_max], [0.0, 0.0], 
    color='red', linewidth=1.0, linestyle='--', 
    label='true parameter', zorder=10
) # 真のパラメータ
ax.plot(
    [mu_min, mu_max], [lambda_truth, lambda_truth], [0.0, 0.0], 
    color='red', linewidth=1.0, linestyle='--', 
    zorder=10
) # 真のパラメータ
ax.contour(
    X=mu_mat, Y=lambda_mat, Z=posterior_dens_mat, offset=0.0, 
    cmap='viridis', vmin=0.0, vmax=dens_max, 
    linewidths=0.8, linestyles=':', 
    zorder=11
) # 事後分布
ax.plot_surface(
    X=mu_mat, Y=lambda_mat, Z=posterior_dens_mat, 
    cmap='viridis', vmin=0.0, vmax=dens_max, 
    alpha=0.5, 
    label='posterior distribution', zorder=12
) # 事後分布
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('density')
ax.legend(prop={'size': 8})
ax.grid(zorder=0)
ax.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)
ax.set_zlim(zmin=0.0, zmax=dens_max)          # (目盛の共通化用)

plt.show()


# %%

### 予測分布(スチューデントのt分布)の計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.95')
mu_s_hat     = m_hat
lambda_s_hat = beta_hat * a_hat / (1.0 + beta_hat) / b_hat
nu_s_hat     = 2.0 * a_hat
#mu_s_hat      = (np.sum(x_n) + beta_hat * m) / (N + beta)
#lambda_s_hat  = (N + beta) * (0.5 * N + a)
#lambda_s_hat /= (N + 1 + beta) * (0.5 * (np.sum(x_n**2) + beta + m**2 - beta_hat * m_hat**2) + beta)
#nu_s_hat      = N + 2 * a


# %%

#### 分布の計算 -----

# 予測分布の確率密度を計算
predict_dens_vec = t.pdf(x=x_vec, df=nu_s_hat, loc=mu_s_hat, scale=1.0/np.sqrt(lambda_s_hat))


# %%

#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl  = f'$N = {N}, '
predict_param_lbl += f'\\mu_{{truth}} = {mu_truth:.3g}, \\lambda_{{truth}} = {lambda_truth:.3g}, '
predict_param_lbl += f'\\hat{{\\mu}}_s = {mu_s_hat:.3g}, \\hat{{\\lambda}}_s = {lambda_s_hat:.3g}, \\hat{{\\nu}}_s = {nu_s_hat:.3g}$'

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


