
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%

# ベイズ推論の可視化 -------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

# 真のパラメータを指定
mu_truth = 25.0

# 既知のパラメータを指定
lmd = 0.01
print(1.0/np.sqrt(lmd))


# %%

### 観測データの生成 -----

# シードを設定(ノートとの対応用)
np.random.seed(86)

# データ数(試行回数)を指定
N = 100

# 観測データを生成
x_n = np.random.normal(loc=mu_truth, scale=1.0/np.sqrt(lmd), size=N)


# %%

### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
k = 4.0
x_size  = 1.0/np.sqrt(lmd) # 標準偏差
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
mu_vec = np.linspace(start=mu_min, stop=mu_max, num=1001)


# %%

### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
m         = 0.0
lambda_mu = 0.001

# 予測分布のパラメータを計算:式(3.62)
mu_s     = m
lambda_s = lmd * lambda_mu / (lmd + lambda_mu)

# 初期値を記録
trace_m_lt         = [m]
trace_lambda_mu_lt = [lambda_mu]
trace_mu_s_lt      = [mu_s]
trace_lambda_s_lt  = [lambda_s]

# ベイズ推論による更新
for n in range(N):

    # 観測データを取得
    x = x_n[n]
    
    # 事後分布のパラメータを更新:式(3.53, 3.54)
    lambda_mu_old = lambda_mu
    lambda_mu += lmd
    m         *= lambda_mu_old
    m         += x * lmd
    m         /= lambda_mu
    
    # 予測分布のパラメータを更新:式(3.62)
    mu_s     = m
    lambda_s = lmd * lambda_mu / (lmd + lambda_mu)
    
    # 更新値を記録
    trace_m_lt.append(m)
    trace_lambda_mu_lt.append(lambda_mu)
    trace_mu_s_lt.append(mu_s)
    trace_lambda_s_lt.append(lambda_s)

    # 動作確認
    print(f'\r{n+1} / {N}', end='', flush=True)


# %%

#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
m         = 0.0
lambda_mu = 0.001

# 事後分布のパラメータを計算:式(3.53, 3.54)
trace_m_lt = np.hstack(
    [m, (np.cumsum(x_n) * lmd + m * lambda_mu) / (np.arange(1, N+1) * lmd + lambda_mu)]
).tolist()
trace_lambda_mu_lt = (
    np.arange(N+1) * lmd + lambda_mu
).tolist()

# 予測分布のパラメータを計算:式(3.62)
trace_mu_s_lt = trace_m_lt.copy()
trace_lambda_s_lt = (
    1.0 / (1.0/lmd + 1.0/np.array(trace_lambda_mu_lt))
).tolist()


# %%

### 推移の作図 -----

#### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_vec = norm.pdf(x=x_vec, loc=mu_truth, scale=1.0/np.sqrt(lmd))

# 事後分布の確率密度を計算
anim_posterior_lt = [
    norm.pdf(x=mu_vec, loc=trace_m_lt[i], scale=1.0/np.sqrt(trace_lambda_mu_lt[i])) for i in range(N+1)
]

# 予測分布の確率密度を計算
anim_predict_lt = [
    norm.pdf(x=x_vec, loc=trace_mu_s_lt[i], scale=1.0/np.sqrt(trace_lambda_s_lt[i])) for i in range(N+1)
]


# %%

#### 事後分布の作図 -----

# 確率密度軸の範囲を設定
u = 0.05
dens_max = np.max(anim_posterior_lt)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)

# 図を初期化
fig, ax = plt.subplots(figsize=(9, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Gaussian distribution', fontsize=20)
ax2 = ax.twiny() # 第2軸の設定用

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()
    ax2.cla()
    
    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    m         = trace_m_lt[i]         # 平均パラメータ
    lambda_mu = trace_lambda_mu_lt[i] # 精度パラメータ
    posterior_dens_vec = anim_posterior_lt[i] # 確率密度
    
    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\mu_{{truth}} = {mu_truth:.2f}, '
    posterior_param_lbl += f'\\hat{{m}} = {m:.2f}, \\hat{{\\lambda}}_{{\\mu}} = {lambda_mu:.5f}$'

    # 事後分布を描画
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', zorder=10
    ) # 真のパラメータ
    ax.plot(
        mu_vec, posterior_dens_vec, 
        color='purple', linewidth=1.0, 
        label='posterior distribution', zorder=11
    ) # 事後分布
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
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('density')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=dens_max)  # 描画範囲を固定
    
    # 第2軸を描画
    ax2.set_xticks(ticks=[mu_truth], labels=['$\mu_{truth}$']) # パラメータラベル
    ax2.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../../figure/gaussian_model/parameter_updates_mean/posterior.mp4', 
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
fig.suptitle('Gaussian distribution', fontsize=20)

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
    mu_s     = trace_mu_s_lt[i]     # 平均パラメータ
    lambda_s = trace_lambda_s_lt[i] # 精度パラメータ
    predict_dens_vec = anim_predict_lt[i] # 確率密度
    
    # 予測分布のラベルを作成
    predict_param_lbl  = f'$N = {n}, '
    predict_param_lbl += f'\\mu_{{truth}} = {mu_truth:.2f}, \\lambda = {lmd:.5f}, '
    predict_param_lbl += f'\\hat{{\\mu}}_{{*}} = {mu_s:.2f}, \\hat{{\\lambda}}_{{*}} = {lambda_s:.5f}$'

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
    filename='../../../figure/gaussian_model/parameter_updates_mean/predict.mp4', 
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

# p(μ)軸の範囲を設定
u = 0.05
posterior_dens_max = np.max(anim_posterior_lt)
posterior_dens_max = np.ceil(posterior_dens_max /u)*u # u単位で切り上げ
print(posterior_dens_max)

# p(x)軸の範囲を設定
u = 0.05
predict_dens_max = np.max(anim_predict_lt)
predict_dens_max = np.ceil(predict_dens_max /u)*u # u単位で切り上げ
print(predict_dens_max)

# %%

# ラベル位置を設定
posterior_loc_x = 0.02
posterior_loc_y = 0.96
predict_loc_x   = 0.02
predict_loc_y   = 0.96

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, 
    figsize=(9, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Bayesian inference', fontsize=20)
axes2 = [ax.twiny() for ax in axes] # 第2軸の設定用
ax2y  = axes[0].twinx() # 第2軸の設定用

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax.cla() for ax in axes2]
    ax2y.cla()
    
    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    m         = trace_m_lt[i]         # 平均パラメータ
    lambda_mu = trace_lambda_mu_lt[i] # 精度パラメータ
    mu_s      = trace_mu_s_lt[i]      # 平均パラメータ
    lambda_s  = trace_lambda_s_lt[i]  # 精度パラメータ
    posterior_dens_vec = anim_posterior_lt[i] # 確率密度
    predict_dens_vec   = anim_predict_lt[i]   # 確率密度
    
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
    model_param_lbl += f'$\\mu_{{truth}} = {mu_truth:.2f}, \\lambda = {lmd:.5f}$\n'
    model_param_lbl += f'$E[x] = \\mu_{{truth}} = {E_x:.2f}$\n'
    model_param_lbl += f'$\\bar{{x}} = {bar_x:.2f}$'

    # 観測データを描画
    ax = axes[0]
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=bar_x, 
        color='hotpink', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 標本平均
    ax.plot(
        x_vec, model_dens_vec, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true model', zorder=12
    ) # 真の分布
    ax.bar(
        x=center_vec, height=obs_dens_vec, 
        width=bin_size, align='center', 
        color='hotpink', alpha=0.5, 
        label='observation data', zorder=13
    ) # 観測データ
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
    ax2 = axes2[0]
    ax2.set_xticks(
        ticks =[E_x, bar_x+1e-10], 
        labels=['$E[x]$', '$\\bar{x}$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    
    freq_max  = dens_max * bin_size*n if n > 0 else 1.0
    dens_vals = ax.get_yticks()        # 確率密度軸目盛を取得
    freq_vals = dens_vals * bin_size*n # 度数軸目盛に変換

    ax2y.set_yticks(
        ticks =freq_vals, 
        labels=[f'{y:.1f}' for y in freq_vals]
    ) # 度数軸目盛
    ax2y.set_ylabel('frequency')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax2y.set_ylim(ymin=0.0, ymax=freq_max) # (目盛の共通化用)

    ##### 事後分布の作図 -----
    
    # 事後分布の期待値を計算
    E_mu = m

    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$\\hat{{m}} = {m:.2f}, \\hat{{\\lambda}}_{{\\mu}} = {lambda_mu:.5f}$\n'
    posterior_param_lbl += f'$E[\\mu] = \\hat{{m}} = {E_mu:.2f}$'

    # 事後分布を描画
    ax = axes[1]
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=E_mu, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 期待値
    ax.plot(
        mu_vec, posterior_dens_vec, 
        color='purple', linewidth=1.0, 
        label='posterior distribution', zorder=12
    ) # 事後分布
    ax.scatter(
        x=x_n[:n], y=np.zeros(shape=n), clip_on=False, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=13
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, clip_on=False, 
        c='hotpink', s=100, 
        zorder=14
    ) # 観測データ

    ax.text(
        x=posterior_loc_x, y=posterior_loc_y, 
        s=posterior_param_lbl, transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('density')
    ax.set_title('Gaussian distribution', loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=posterior_dens_max) # 描画範囲を固定
    
    # 第2軸を描画
    ax2 = axes2[1]
    ax2.set_xticks(
        ticks =[mu_truth, E_mu+1e-10], 
        labels=['$\\mu_{{truth}}$', '$E[\\mu]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)

    ##### 予測分布の作図 -----

    # 予測分布の期待値を計算
    E_x = mu_s
    
    # 予測分布のラベルを作成
    predict_param_lbl  = f'$\\hat{{\\mu}}_{{*}} = {mu_s:.2f}, \\hat{{\\lambda}}_{{*}} = {lambda_s:.5f}$\n'
    predict_param_lbl += f'$E[x] = \\hat{{\\mu}}_{{*}} = {E_x:.2f}$'

    # 予測分布を描画
    ax = axes[2]
    ax.axvline(
        x=mu_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=E_x, 
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
    ax.set_title('Gaussian distribution', loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=predict_dens_max) # 描画範囲を固定

    # 第2軸を描画
    ax2 = axes2[2]
    ax2.set_xticks(
        ticks =[mu_truth, E_x+1e-10], 
        labels=['$\\mu_{{truth}}$', '$E[x]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../../figure/gaussian_model/parameter_updates_mean/observation.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


