
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.3
# 平均と精度が未知の場合
# ベイズ推論
# 学習推移の可視化


# %% 

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm


# %%

# ベイズ推論の可視化 -------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

# 真のパラメータを指定
mu_truth     = 5.0
lambda_truth = 0.25

# 標準偏差パラメータに変換
sigma_truth = 1.0/np.sqrt(lambda_truth) # (処理の効率化用)
print(sigma_truth)


# %%

### 観測データの生成 -----

# シードを設定(ノートとの対応用)
np.random.seed(86)

# データ数(試行回数)を指定
N = 100

# 観測データを生成
x_n = np.random.normal(loc=mu_truth, scale=sigma_truth, size=N)


# %%

### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
k = 4.0
x_size  = sigma_truth # 標準偏差
x_size *= k # 定数倍
x_size  = max(x_size, (x_n-mu_truth).max()) # サンプルと比較
x_size  = np.ceil(x_size /u)*u # u単位で切り上げ
x_min   = mu_truth - x_size
x_max   = mu_truth + x_size
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# μ軸の範囲を設定
mu_min = x_min # (固定)
mu_max = x_max # (固定)
print('μ size:', mu_min, mu_max)

# μ軸の値を作成
mu_vec = np.linspace(start=mu_min, stop=mu_max, num=501)


# λ軸の範囲を設定
lambda_min = 0.0
#lambda_max = 1.0
u = 0.6
k = 2.5
lambda_max = lambda_truth # 真値
lambda_max *= k # 定数倍
lambda_max = np.ceil(lambda_max /u)*u # u単位で切り上げ
print('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec = np.linspace(start=lambda_min, stop=lambda_max, num=501)


# 格子点を作成
mu_mat, lambda_mat = np.meshgrid(mu_vec, lambda_vec)


# σ軸の範囲を設定
sigma_min = x_min - mu_truth # (固定)
sigma_max = x_max - mu_truth # (固定)
print('σ size:', sigma_min, sigma_max)

# σ軸の値を作成
sigma_vec = np.linspace(start=sigma_min, stop=sigma_max, num=1001)


# %%

### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
m    = 0.0
beta = 1.0
a    = 1.0
b    = 1.0

# 初期値による予測分布のパラメータを計算:式(3.95)
mu_s     = m
lambda_s = beta * a / (1.0 + beta) / b
nu_s     = 2.0 * a

# 初期値を記録
trace_m_lt        = [m]
trace_beta_lt     = [beta]
trace_a_lt        = [a]
trace_b_lt        = [b]
trace_mu_s_lt     = [mu_s]
trace_lambda_s_lt = [lambda_s]
trace_nu_s_lt     = [nu_s]

# ベイズ推論による更新
for n in range(N):

    # 観測データを取得
    x = x_n[n]
    
    # μの事後分布のパラメータを更新:式(3.83)
    old_beta = beta
    old_m    = m
    beta += 1.0
    m     = (x_n[n] + old_beta * old_m) / beta
    
    # λの事後分布のパラメータを更新:式(3.88)
    a += 0.5
    b += 0.5 * (x_n[n]**2 + old_beta * old_m**2 - beta * m**2)
    
    # 予測分布のパラメータを更新:式(3.95)
    mu_s      = m
    lambda_s  = beta / (1.0 + beta) * a / b
    nu_s     += 1.0
    
    # 更新値を記録
    trace_m_lt.append(m)
    trace_beta_lt.append(beta)
    trace_a_lt.append(a)
    trace_b_lt.append(b)
    trace_mu_s_lt.append(mu_s)
    trace_lambda_s_lt.append(lambda_s)
    trace_nu_s_lt.append(nu_s)

    # 動作確認
    print(f'{n+1} / {N}')


# %%

#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
m    = 0.0
beta = 1.0
a    = 1.0
b    = 1.0

# μの事後分布のパラメータを計算:式(3.83)
trace_m_lt = np.hstack(
    [m, (np.cumsum(x_n) + beta * m) / (np.arange(1, N+1) + beta)]
).tolist()
trace_beta_lt = (
    np.arange(N+1) + beta
).tolist()

# λの事後分布のパラメータを計算:式(3.88)
trace_a_lt = (
    0.5 * np.arange(N+1) + a
).tolist()
trace_b_lt = (
    0.5 * (np.cumsum(np.hstack([0.0, x_n])**2) + beta * m**2 - np.array(trace_beta_lt) * np.array(trace_m_lt)**2) + b
).tolist()

# 予測分布のパラメータを計算:式(3.95)
trace_mu_s_lt = trace_m_lt.copy()
trace_lambda_s_lt  = (
    np.array(trace_beta_lt) / (1.0 + np.array(trace_beta_lt)) * np.array(trace_a_lt) / np.array(trace_b_lt)
).tolist()
trace_nu_s_lt = (
    np.arange(N+1) + 2.0 * a
).tolist()


# %%

### 推移の作図 -----

#### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_vec = norm.pdf(x=x_vec, loc=mu_truth, scale=sigma_truth)

# 事後分布の確率密度を計算
anim_posterior_mu_lt = [
    norm.pdf(x=mu_mat, loc=trace_m_lt[i], scale=1.0/np.sqrt(trace_beta_lt[i]*lambda_mat)) for i in range(N+1)
]
anim_posterior_lambda_lt = [
    gamma.pdf(x=lambda_mat, a=trace_a_lt[i], scale=1.0/trace_b_lt[i]) for i in range(N+1)
]
anim_posterior_lt = [
    anim_posterior_mu_lt[i] * anim_posterior_lambda_lt[i] for i in range(N+1)
]

# 予測分布の確率密度を計算
anim_predict_lt = [
    t.pdf(x=x_vec, df=trace_nu_s_lt[i], loc=trace_mu_s_lt[i], scale=1.0/np.sqrt(trace_lambda_s_lt[i])) for i in range(N+1)
]


# %%

#### 事後分布の作図 -----

# 確率密度軸の範囲を設定
u = 0.5
dens_max = np.max(anim_posterior_lt)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)

# 等高線を設定
level_num = 21
dens_vals = np.linspace(start=0.0, stop=dens_max, num=level_num)
print(dens_vals)

# %%

## 等高線図

# 図を初期化
fig, ax = plt.subplots(figsize=(9, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Gaussian-Gamma distribution', fontsize=20)
ax2x = ax.twiny() # 第2軸の設定用
ax2y = ax.twinx() # 第2軸の設定用
cs = ax.contourf(
    mu_mat, lambda_mat, np.zeros_like(mu_mat), 
    cmap='viridis', vmin=0.0, vmax=dens_max, levels=dens_vals
) # (カラーバー表示用のダミー)
fig.colorbar(cs, ax=ax, shrink=1.0, label='density')

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()
    ax2x.cla()
    ax2y.cla()
    
    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    m    = trace_a_lt[i]    # 平均パラメータ
    beta = trace_beta_lt[i] # 係数パラメータ
    a    = trace_a_lt[i]    # 形状パラメータ
    b    = trace_b_lt[i]    # 尺度パラメータ
    posterior_dens_mat = anim_posterior_lt[i] # 確率密度
    
    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\mu_{{truth}} = {mu_truth:.2f}, \\lambda_{{truth}} = {lambda_truth:.2f}, '
    posterior_param_lbl += f'\\hat{{m}} = {m:.1f}, \\hat{{\\beta}} = {beta:.1f}, \\hat{{a}} = {a:.1f}, \\hat{{b}} = {b:.1f}$'

    # 事後分布を描画
    ax.bar(
        x=mu_truth, height=0.0, 
        facecolor=cm.viridis(X=0.0, alpha=0.5), edgecolor='none', 
        label='posterior distribution', zorder=10
    ) # (凡例表示用のダミー)
    ax.contourf(
        mu_mat, lambda_mat, posterior_dens_mat, 
        cmap='viridis', vmin=0.0, vmax=dens_max, levels=dens_vals, 
        alpha=0.5, 
        zorder=11
    ) # 事後分布
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', zorder=12
    ) # 真のパラメータ
    ax.axhline(
        y=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.scatter(
        x=x_n[:n], y=(sigma_truth/(x_n[:n]-mu_truth))**2, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=13
    ) # 観測データ
    ax.scatter(
        x=x, y=(sigma_truth/(x-mu_truth))**2, 
        c='hotpink', s=100, 
        label='observation data', zorder=14
    ) # 観測データ
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
    ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)
    
    # 第2軸を描画
    ax2x.set_xticks(ticks=[mu_truth], labels=['$\mu_{truth}$']) # パラメータラベル
    ax2y.set_yticks(ticks=[lambda_truth], labels=['$\lambda_{truth}$']) # パラメータラベル
    ax2x.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
    ax2y.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../../figure/gaussian_model/parameter_updates_mean_precision/posterior_2d.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

## 曲面図

# 図を初期化
fig, ax = plt.subplots(
    figsize=(9, 6), dpi=100, facecolor='white', 
    constrained_layout=True, subplot_kw={'projection': '3d'}
)
fig.suptitle('Gaussian-Gamma distribution', fontsize=20)
cs = ax.contour(
    X=mu_mat, Y=lambda_mat, Z=np.zeros_like(mu_mat), offset=0.0, 
    cmap='viridis', vmin=0.0, vmax=dens_max, levels=dens_vals
) # (カラーバー表示用のダミー)
fig.colorbar(cs, ax=ax, shrink=1.0, label='density')

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()
    ax2x.cla()
    ax2y.cla()
    
    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    m    = trace_a_lt[i]    # 平均パラメータ
    beta = trace_beta_lt[i] # 係数パラメータ
    a    = trace_a_lt[i]    # 形状パラメータ
    b    = trace_b_lt[i]    # 尺度パラメータ
    posterior_dens_mat = anim_posterior_lt[i] # 確率密度
    
    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\mu_{{truth}} = {mu_truth:.2f}, \\lambda_{{truth}} = {lambda_truth:.2f}, '
    posterior_param_lbl += f'\\hat{{m}} = {m:.1f}, \\hat{{\\beta}} = {beta:.1f}, \\hat{{a}} = {a:.1f}, \\hat{{b}} = {b:.1f}$'

    # 事後分布を描画
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
        cmap='viridis', vmin=0.0, vmax=dens_max, levels=dens_vals, 
        linewidths=0.5, 
        zorder=11
    ) # 事後分布
    tmp_val_n  = (sigma_truth/(x_n[:n]-mu_truth))**2
    tmp_bool_n = tmp_val_n <= lambda_max
    ax.scatter(
        xs=x_n[:n][tmp_bool_n], ys=tmp_val_n[tmp_bool_n], zs=np.zeros(shape=np.sum(tmp_bool_n)), 
        c='hotpink', alpha=0.33, s=25, 
        zorder=12
    ) # 観測データ
    tmp_val = (sigma_truth/(x-mu_truth))**2
    tmp_val = tmp_val if tmp_val <= lambda_max else np.nan
    ax.scatter(
        xs=x, ys=tmp_val, zs=0.0, 
        c='hotpink', s=100, 
        label='observation data', zorder=13
    ) # 観測データ
    ax.plot_surface(
        X=mu_mat, Y=lambda_mat, Z=posterior_dens_mat, 
        cmap='viridis', vmin=0.0, vmax=dens_max, 
        alpha=0.5, 
        label='posterior distribution', zorder=14
    ) # 事後分布
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('density')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
    ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)
    ax.set_zlim(zmin=0.0, zmax=dens_max)          # 描画範囲を固定

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../../figure/gaussian_model/parameter_updates_mean_precision/posterior_3d.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

#### 予測分布の作図 -----

# 確率密度軸の範囲を設定
u = 0.05
dens_max = np.max(anim_predict_lt)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)

# 図を初期化
fig, ax = plt.subplots(figsize=(9, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle("Student's t Distribution", fontsize=20)

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()
    
    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    mu_s     = trace_mu_s_lt[i]     # 位置パラメータ
    lambda_s = trace_lambda_s_lt[i] # 尺度パラメータ
    nu_s     = trace_nu_s_lt[i]     # 自由度パラメータ
    predict_dens_vec = anim_predict_lt[i] # 確率密度
    
    # 予測分布のラベルを作成
    predict_param_lbl  = f'$N = {n}, '
    predict_param_lbl += f'\\mu_{{truth}} = {mu_truth:.2f}, \\lambda_{{truth}} = {lambda_truth:.5f}, '
    predict_param_lbl += f'\\hat{{\\mu}}_{{s}} = {mu_s:.2f}, \\hat{{\\lambda}}_{{s}} = {lambda_s:.5f}, \\hat{{\\nu}}_{{s}} = {nu_s:.1f}$'

    # 予測分布を描画
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
    ax.scatter(
        x=x_n[:n], y=np.zeros(shape=n), clip_on=False, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=12
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, clip_on=False, 
        c='hotpink', s=100, 
        label='observation data', zorder=13
    ) # 観測データ
    ax.set_xlabel('$x$')
    ax.set_ylabel('density')
    ax.set_title(predict_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min, xmax=x_max)  # 描画範囲を固定
    ax.set_ylim(ymin=0.0, ymax=dens_max) # 描画範囲を固定

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../../figure/gaussian_model/parameter_updates_mean_precision/predict.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

#### 観測データと分布の関係 -----

# 階級数を指定
bin_num = 40

# 階級値を作成
bin_size = (x_max - x_min) / bin_num # 階級幅
bin_min  = x_min - 0.5*bin_size # 境界値の最小値
bin_max  = x_max + 0.5*bin_size # 境界値の最大値
_, bin_vec = np.histogram(a=x_n, bins=bin_num+1, range=(bin_min, bin_max), density=True) # 境界値
center_vec = bin_vec[:-1] + 0.5*bin_size # 階級値

# p(μ,λ)軸の範囲を設定
u = 0.1
posterior_dens_max = np.max(anim_posterior_lt)
posterior_dens_max = np.ceil(posterior_dens_max /u)*u # u単位で切り上げ
print(posterior_dens_max)

# 等高線を設定
level_num = 21
posterior_dens_vals = np.linspace(start=0.0, stop=posterior_dens_max, num=level_num)
print(posterior_dens_vals)

# p(x)軸の範囲を設定
u = 0.05
predict_dens_max = np.max(anim_predict_lt)
predict_dens_max = np.ceil(predict_dens_max /u)*u # u単位で切り上げ
print(predict_dens_max)

# %%

# ラベル位置を設定
posterior_loc_x = 0.02
posterior_loc_y = 0.98
predict_loc_x   = 0.02
predict_loc_y   = 0.96

# 図を初期化
fig, axes = plt.subplots(
    nrows=4, ncols=2, 
    figsize=(15, 20), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Bayesian inference', fontsize=20)
axes2x = [ax.twiny() for ax in [axes[1, 0], axes[2, 0], axes[3, 0], axes[3, 1]]] # 第2軸の設定用
axes2y = [ax.twinx() for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]] # 第2軸の設定用

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes.flatten()]
    [ax.cla() for ax in axes2x]
    [ax.cla() for ax in axes2y]

    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    m        = trace_m_lt[i]        # 平均パラメータ
    beta     = trace_beta_lt[i]     # 係数パラメータ
    a        = trace_a_lt[i]        # 形状パラメータ
    b        = trace_b_lt[i]        # 尺度パラメータ
    mu_s     = trace_mu_s_lt[i]     # 位置パラメータ
    lambda_s = trace_lambda_s_lt[i] # 尺度パラメータ
    nu_s     = trace_nu_s_lt[i]     # 自由度パラメータ
    posterior_dens_mat = anim_posterior_lt[i] # 確率密度
    predict_dens_vec   = anim_predict_lt[i]   # 確率密度

    ##### 軸変換の作図：(σ to σ) -----

    # 恒等関数を描画
    ax  = axes[0, 0]
    ax.plot(
        sigma_vec, sigma_vec, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 恒等関数
    ax.vlines(
        x=sigma_truth, ymin=sigma_min, ymax=sigma_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ
    ax.hlines(
        y=sigma_truth, xmin=sigma_truth, xmax=sigma_max, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ

    ax.set_xlabel('$\\sigma$')
    ax.set_ylabel('$\\sigma$')
    ax.grid(zorder=0)
    ax.set_xlim(xmin=sigma_min, xmax=sigma_max) # (目盛の共通化用)
    ax.set_ylim(ymin=sigma_min, ymax=sigma_max) # (目盛の共通化用)

    # 第2軸を描画
    ax2y = axes2y[0]
    ax2y.set_yticks(ticks=[sigma_truth], labels=['$\\sigma_{truth}$']) # パラメータラベル
    ax2y.set_ylim(ymin=sigma_min, ymax=sigma_max) # (目盛の共通化用)

    ##### 軸変換の作図：(σ to λ) -----

    # (警告文の回避用)
    tmp_sigma_vec = sigma_vec[sigma_vec > 0.0]

    # 変換曲線を描画
    ax = axes[0, 1]
    ax.plot(
        1.0/tmp_sigma_vec**2, tmp_sigma_vec, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 変換曲線
    ax.hlines(
        y=sigma_truth, xmin=lambda_min, xmax=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ
    ax.vlines(
        x=lambda_truth, ymin=sigma_min, ymax=sigma_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ

    ax.set_xlabel('$\\lambda = \\frac{1}{\\sigma^2}$')
    ax.set_ylabel('$\\sigma = \\frac{1}{\\sqrt{\\lambda}}$')
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)
    ax.set_ylim(ymin=sigma_min, ymax=sigma_max)   # (目盛の共通化用)

    ##### 観測データの作図 -----

    # 生成分布の期待値を計算
    E_x = mu_truth

    # 観測データの標本平均を計算
    bar_x = np.mean(x_n[:n]) if n > 0 else np.nan

    # 観測データを集計
    if n > 0:
        obs_dens_vec, _ = np.histogram(a=x_n[:n], bins=bin_num+1, range=(bin_min, bin_max), density=True)
    else:
        obs_dens_vec = np.zeros(shape=bin_num+1) # (警告文の回避用)

    # 生成分布のラベルを作成
    model_param_lbl  = f'$N = {n}$\n'
    model_param_lbl += f'$\\mu_{{truth}} = {mu_truth:.1f}, \\lambda_{{truth}} = {lambda_truth:.3f}$\n'
    model_param_lbl += f'$E[x] = \\mu = {E_x:.2f}$\n'
    model_param_lbl += f'$\\bar{{x}} = {bar_x:.2f}$'

    # 観測データを描画
    ax = axes[1, 0]
    ax.axvline(
        x=mu_truth+sigma_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ
    ax.axvline(
        x=bar_x, 
        color='hotpink', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 標本平均
    ax.plot(
        x_vec, model_dens_vec, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true model', zorder=13
    ) # 真の分布
    ax.bar(
        x=center_vec, height=obs_dens_vec, 
        width=bin_size, align='center', 
        color='hotpink', alpha=0.5, 
        label='observation data', zorder=14
    ) # 観測データ
    ax.scatter(
        x=x_n[:n], y=np.zeros(shape=n), clip_on=False, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=15
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, clip_on=False, 
        c='hotpink', s=100, 
        zorder=16
    ) # 観測データ

    ax.text(
        x=predict_loc_x, y=predict_loc_y, 
        s=model_param_lbl, transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$x$')
    ax.set_ylabel('density')
    ax.set_title('Gaussian distribution', loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min, xmax=x_max)          # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=predict_dens_max) # (目盛の共通化用)

    # 第2軸を描画
    ax2x = axes2x[0]
    ax2x.set_xticks(
        ticks =[mu_truth+sigma_truth, E_x, bar_x+1e-10], 
        labels=['$\\mu_{truth} + \\sigma_{truth}$', '$E[x]$', '$\\bar{x}$']
    ) # パラメータラベル
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)

    freq_max      = predict_dens_max * bin_size*n if n > 0 else 1.0
    obs_dens_vals = ax.get_yticks()            # 確率密度軸目盛を取得
    freq_vals     = obs_dens_vals * bin_size*n # 度数軸目盛に変換

    ax2y = axes2y[1]
    ax2y.set_yticks(
        ticks =freq_vals, 
        labels=[f'{y:.1f}' for y in freq_vals]
    ) # 度数軸目盛
    ax2y.set_ylabel('frequency')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax2y.set_ylim(ymin=0.0, ymax=freq_max) # (目盛の共通化用)

    ##### λ軸の作図 -----

    # 変換曲線を描画
    ax = axes[1, 1]
    ax.axvline(
        x=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ

    ax.set_xlabel('$\\lambda$')
    ax.set_yticks(ticks=[]) # 目盛を非表示
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)

    ##### 事後分布の作図 -----

    # 事後分布の期待値を計算
    E_mu     = m
    E_lambda = a / b

    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$\\hat{{m}} = {m:.2f}, \\hat{{\\beta}} = {beta:.1f}, '
    posterior_param_lbl += f'\\hat{{a}} = {a:.1f}, \\hat{{b}} = {b:.1f}$\n'
    posterior_param_lbl += f'$E[\\mu] = \\hat{{m}} = {E_mu:.2f}, '
    posterior_param_lbl += f'E[\\lambda] = \\frac{{\\hat{{a}}}}{{\\hat{{b}}}} = {E_lambda:.3f}$'

    # 事後分布を描画
    ax = axes[2, 0]
    ax.bar(
        x=mu_truth, height=0.0, 
        facecolor=cm.viridis(X=0.0, alpha=0.5), edgecolor='none', 
        label='posterior distribution', zorder=10
    ) # (凡例表示用のダミー)
    ax.contourf(
        mu_mat, lambda_mat, posterior_dens_mat, 
        cmap='viridis', vmin=0.0, vmax=posterior_dens_max, levels=posterior_dens_vals, 
        alpha=0.5, 
        zorder=11
    ) # 事後分布
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', zorder=12
    ) # 真のパラメータ
    ax.axhline(
        y=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.axvline(
        x=E_mu, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値
    ax.axhline(
        y=E_lambda, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値
    ax.scatter(
        x=x_n[:n], y=(sigma_truth/(x_n[:n]-mu_truth))**2, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=14
    ) # 観測データ
    ax.scatter(
        x=x, y=(sigma_truth/(x-mu_truth))**2, 
        c='hotpink', s=100, 
        zorder=15
    ) # 観測データ

    ax.text(
        x=posterior_loc_x, y=posterior_loc_y, 
        s=posterior_param_lbl, transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$\\mu$')
    ax.set_ylabel('$\\lambda$')
    ax.set_title('Gaussian-Gamma distribution', loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=mu_min, xmax=mu_max)         # (目盛の共通化用)
    ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)

    # 第2軸を描画
    ax2x = axes2x[1]
    ax2x.set_xticks(
        ticks =[mu_truth, E_mu+1e-10], 
        labels=['$\\mu_{{truth}}$', '$E[\\mu]$']
    ) # パラメータラベル
    ax2x.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)

    ax2y = axes2y[2]
    ax2y.set_yticks(
        ticks =[lambda_truth, E_lambda+1e-10], 
        labels=['$\\lambda_{{truth}}$', '$E[\\lambda]$']
    ) # パラメータラベル
    ax2y.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)

    ##### 軸変換の作図：(λ to λ) -----

    # 恒等関数を描画
    ax  = axes[2, 1]
    ax.plot(
        lambda_vec, lambda_vec, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 恒等関数
    ax.axvline(
        x=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ
    ax.hlines(
        y=lambda_truth, xmin=lambda_min, xmax=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 真のパラメータ
    ax.hlines(
        y=E_lambda, xmin=lambda_min, xmax=E_lambda, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.vlines(
        x=E_lambda, ymin=lambda_min, ymax=E_lambda, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 期待値

    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('$\\lambda$')
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)
    ax.set_ylim(ymin=lambda_min, ymax=lambda_max) # (目盛の共通化用)

    ##### 予測分布の作図 -----

    # 予測分布の期待値を計算
    E_x = mu_s

    # 予測分布のラベルを作成
    predict_param_lbl  = f'$\\hat{{\\mu}}_{{s}} = {mu_s:.2f}, \\hat{{\\lambda}}_{{s}} = {lambda_s:.3f}, \\hat{{\\nu}}_{{s}} = {nu_s:.1f}$\n'
    predict_param_lbl += f'$E[x] = \\hat{{\\mu}}_{{s}} = {E_x:.2f}$'

    # 予測分布を描画
    ax = axes[3, 0]
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axhline(
        y=norm.pdf(x=mu_truth, loc=mu_truth, scale=sigma_truth), 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=E_x, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 期待値
    ax.axhline(
        y=t.pdf(x=E_x, df=nu_s, loc=E_x, scale=1.0/np.sqrt(lambda_s)), 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 期待値
    ax.plot(
        x_vec, model_dens_vec, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true model', zorder=12
    ) # 真の分布
    ax.plot(
        x_vec, predict_dens_vec, 
        color='purple', linewidth=1.0, 
        label='predict distribution', zorder=13
    ) # 予測分布
    ax.scatter(
        x=x_n[:n], y=np.zeros(shape=n), clip_on=False, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=14
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, clip_on=False, 
        c='hotpink', s=100, 
        zorder=15
    ) # 観測データ

    ax.text(
        x=predict_loc_x, y=predict_loc_y, 
        s=predict_param_lbl, transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$x$')
    ax.set_ylabel('density')
    ax.set_title("Student's t Distribution", loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min, xmax=x_max)          # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=predict_dens_max) # (目盛の共通化用)

    # 第2軸を描画
    ax2 = axes2x[2]
    ax2.set_xticks(
        ticks =[mu_truth, E_x+1e-10], 
        labels=['$\\mu_{truth}$', '$E[x]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)

    ##### 軸変換の作図：(λ to p(λ)) -----

    # (警告文の回避用)
    tmp_lambda_vec = lambda_vec[lambda_vec > 0.0]

    # 確率密度を計算
    tmp_N_dens_vec = norm.pdf(x=mu_truth, loc=mu_truth, scale=1.0/np.sqrt(tmp_lambda_vec))
    tmp_N_dens_val = norm.pdf(x=mu_truth, loc=mu_truth, scale=sigma_truth)
    tmp_t_dens_vec = t.pdf(x=mu_s, df=nu_s, loc=mu_s, scale=1.0/np.sqrt(tmp_lambda_vec))
    tmp_t_dens_val = t.pdf(x=mu_s, df=nu_s, loc=mu_s, scale=1.0/np.sqrt(lambda_s))

    # 変換曲線を描画
    ax = axes[3, 1]
    ax.plot(
        tmp_lambda_vec, tmp_N_dens_vec, 
        color='black', linewidth=1.0, linestyle='--', 
        label='$N(x = \\mu \\mid \\mu, \\lambda^{-1})$', zorder=10
    ) # 変換曲線:ガウス分布
    ax.plot(
        tmp_lambda_vec, tmp_t_dens_vec, 
        color='black', linewidth=1.0, 
        label='$St(x = \\hat{\\mu}_s \\mid \\hat{\\mu}_s, \\lambda, \\hat{\\nu}_s)$', zorder=11
    ) # 変換曲線:t分布
    ax.vlines(
        x=lambda_truth, ymin=tmp_N_dens_val, ymax=lambda_max, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.hlines(
        y=tmp_N_dens_val, xmin=lambda_min, xmax=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.vlines(
        x=lambda_s, ymin=tmp_t_dens_val, ymax=lambda_max, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値
    ax.hlines(
        y=tmp_t_dens_val, xmin=lambda_min, xmax=lambda_s, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値

    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('density')
    ax.legend(loc='lower right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=predict_dens_max)  # (目盛の共通化用)

    # 第2軸を描画
    ax2x = axes2x[3]
    ax2x.set_xticks(
        ticks =[lambda_truth, lambda_s+1e-10], 
        labels=['$\\lambda_{truth}$', '$\\hat{\\lambda}_s$']
    ) # パラメータラベル
    ax2x.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../../figure/gaussian_model/parameter_updates_mean_precision/_observation.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


