
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.2
# 精度が未知の場合
# ベイズ推論
# 学習推移の可視化


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%

# ベイズ推論の可視化 -------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

# 既知のラメータを指定
mu = 5.0

# 真のパラメータを指定
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
x_n = np.random.normal(loc=mu, scale=sigma_truth, size=N)


# %%

### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
x_size  = sigma_truth # 基準値を指定
x_size *= 4.0 # 倍率を指定
x_size  = max(x_size, (x_n-mu).max()) # サンプルと比較
x_size  = np.ceil(x_size /u)*u # u単位で切り上げ
x_min   = mu - x_size
x_max   = mu + x_size
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# λ軸の範囲を設定
lambda_min = 0.0
#lambda_max = 1.0
u = 0.5
lambda_max = lambda_truth # 基準値を指定
lambda_max *= 5.0 # 倍率を指定
lambda_max = np.ceil(lambda_max /u)*u # u単位で切り上げ
print('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec = np.linspace(start=lambda_min, stop=lambda_max, num=1001)


# σ軸の範囲を設定
sigma_min = x_min - mu # (固定)
sigma_max = x_max - mu # (固定)
print('σ size:', sigma_min, sigma_max)

# σ軸の値を作成
sigma_vec = np.linspace(start=sigma_min, stop=sigma_max, num=1001)


# %%

### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
a = 1.0
b = 1.0

# 予測分布のパラメータを計算:式(3.79)
mu_s     = mu
lambda_s = a / b
nu_s     = 2.0 * a

# 初期値を記録
trace_a_lt        = [a]
trace_b_lt        = [b]
trace_mu_s_lt     = [mu_s]
trace_lambda_s_lt = [lambda_s]
trace_nu_s_lt     = [nu_s]

# ベイズ推論による更新
for n in range(N):

    # 観測データを取得
    x = x_n[n]
    
    # 事後分布のパラメータを計算:式(3.69)
    a += 0.5
    b += 0.5 * (x_n[n] - mu)**2

    # 予測分布のパラメータを計算:式(3.79)
    mu_s     = mu
    lambda_s = a / b
    nu_s     = 2.0 * a
    
    # 更新値を記録
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
a = 1.0
b = 1.0

# 事後分布のパラメータを計算:式(3.69)
trace_a_lt = (
    0.5 * np.arange(N+1) + a
).tolist()
trace_b_lt = np.hstack(
    [b, 0.5 * np.cumsum((x_n - mu)**2) + b]
).tolist()

# 予測分布のパラメータを計算:式(3.79)
trace_mu_s_lt = np.tile(mu, reps=N+1).tolist()
trace_lambda_s_lt = np.hstack(
    [a/b, (np.arange(1, N+1) + 2.0 * a) / (np.cumsum((x_n - mu)**2) + 2.0 * b)]
).tolist()
trace_nu_s_lt = (
    np.arange(N+1) + 2.0 * a
).tolist()


# %%

### 推移の作図 -----

#### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_vec = norm.pdf(x=x_vec, loc=mu, scale=sigma_truth)

# 事後分布の確率密度を計算
anim_posterior_lt = [
    gamma.pdf(x=lambda_vec, a=trace_a_lt[i], scale=1.0/trace_b_lt[i]) for i in range(N+1)
]

# 予測分布の確率密度を計算
anim_predict_lt = [
    t.pdf(x=x_vec, df=trace_nu_s_lt[i], loc=trace_mu_s_lt[i], scale=1.0/np.sqrt(trace_lambda_s_lt[i])) for i in range(N+1)
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
fig.suptitle('Gamma distribution', fontsize=20)
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
    a = trace_a_lt[i] # 形状パラメータ
    b = trace_b_lt[i] # 尺度パラメータ
    posterior_dens_vec = anim_posterior_lt[i] # 確率密度
    
    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.2f}, '
    posterior_param_lbl += f'\\hat{{a}} = {a:.1f}, \\hat{{b}} = {b:.1f}$'

    # 事後分布を描画
    ax.axvline(
        x=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', zorder=10
    ) # 真のパラメータ
    ax.plot(
        lambda_vec, posterior_dens_vec, 
        color='purple', linewidth=1.0, 
        label='posterior distribution', zorder=11
    ) # 事後分布
    ax.scatter(
        x=(sigma_truth/(x_n[:n]-mu))**2, y=np.zeros(shape=n), 
        c='hotpink', alpha=0.33, s=25, 
        clip_on=False, zorder=12
    ) # 観測データ
    ax.scatter(
        x=(sigma_truth/(x-mu))**2, y=0.0, 
        c='hotpink', s=100, 
        label='observation data', clip_on=False, zorder=13
    ) # 観測データ
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('density')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=dens_max)          # 描画範囲を固定
    
    # 第2軸を描画
    ax2.set_xticks(ticks=[lambda_truth], labels=['$\lambda_{truth}$']) # パラメータラベル
    ax2.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../figure/gaussian_model/parameter_updates_precision/posterior.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
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
    predict_param_lbl += f'\\ = {mu:.2f}, \\lambda_{{truth}} = {lambda_truth:.5f}, '
    predict_param_lbl += f'\\mu_{{s}} = {mu_s:.2f}, \\hat{{\\lambda}}_{{s}} = {lambda_s:.5f}, \\hat{{\\nu}}_{{s}} = {nu_s:.1f}$'

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
        x=x_n[:n], y=np.zeros(shape=n), 
        c='hotpink', alpha=0.33, s=25, 
        clip_on=False, zorder=12
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, 
        c='hotpink', s=100, 
        label='observation data', clip_on=False, zorder=12
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
    filename='../../figure/gaussian_model/parameter_updates_precision/predict.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
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

# ラベル位置を設定
posterior_loc_x = 0.02
posterior_loc_y = 0.98
predict_loc_x   = 0.02
predict_loc_y   = 0.96

# 確率密度軸の範囲を設定
u = 0.05
posterior_dens_max = np.max(anim_posterior_lt)
posterior_dens_max = np.ceil(posterior_dens_max /u)*u # u単位で切り上げ
predict_dens_max   = np.max(anim_predict_lt)
predict_dens_max   = np.ceil(predict_dens_max /u)*u # u単位で切り上げ
print(posterior_dens_max)
print(predict_dens_max)

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=2, 
    figsize=(15, 15), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Bayesian inference', fontsize=20)
axes2x = [ax.twiny() for ax in [axes[1, 0], axes[1, 1], axes[2, 0]]] # 第2軸の設定用
axes2y = [ax.twinx() for ax in [axes[0, 0], axes[1, 0]]]             # 第2軸の設定用

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
    a        = trace_a_lt[i] # 形状パラメータ
    b        = trace_b_lt[i] # 尺度パラメータ
    mu_s     = trace_mu_s_lt[i]     # 位置パラメータ
    lambda_s = trace_lambda_s_lt[i] # 尺度パラメータ
    nu_s     = trace_nu_s_lt[i]     # 自由度パラメータ
    sigma_truth = 1.0/np.sqrt(lambda_truth)
    posterior_dens_vec = anim_posterior_lt[i] # 確率密度
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
    E_x = mu

    # 観測データの標本平均を計算
    bar_x = np.mean(x_n[:n]) if n > 0 else np.nan

    # 観測データを集計
    if n > 0:
        obs_dens_vec, _ = np.histogram(a=x_n[:n], bins=bin_num+1, range=(bin_min, bin_max), density=True)
    else:
        obs_dens_vec = np.zeros(shape=bin_num+1) # (警告文の回避用)

    # 生成分布のラベルを作成
    model_param_lbl  = f'$N = {n}$\n'
    model_param_lbl += f'$\\mu = {mu:.1f}, \\lambda_{{truth}} = {lambda_truth:.3f}$\n'
    model_param_lbl += f'$E[x] = \\mu = {E_x:.2f}$\n'
    model_param_lbl += f'$\\bar{{x}} = {bar_x:.2f}$'

    # 観測データを描画
    ax = axes[1, 0]
    ax.axvline(
        x=mu+sigma_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=mu, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 既知のパラメータ
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
        x=x_n[:n], y=np.zeros(shape=n), 
        c='hotpink', alpha=0.33, s=25, 
        clip_on=False, zorder=15
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, 
        c='hotpink', s=100, 
        clip_on=False, zorder=16
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
    ax2 = axes2x[0]
    ax2.set_xticks(
        ticks =[mu+sigma_truth, E_x, bar_x+1e-10], 
        labels=['$\\mu + \\sigma_{truth}$', '$E[x]$', '$\\bar{x}$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    
    freq_max  = predict_dens_max * bin_size*n if n > 0 else 1.0
    dens_vals = ax.get_yticks()        # 確率密度軸目盛を取得
    freq_vals = dens_vals * bin_size*n # 度数軸目盛に変換

    ax2y = axes2y[1]
    ax2y.set_yticks(
        ticks =freq_vals, 
        labels=[f'{y:.1f}' for y in freq_vals]
    ) # 度数軸目盛
    ax2y.set_ylabel('frequency')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax2y.set_ylim(ymin=0.0, ymax=freq_max) # (目盛の共通化用)

    ##### 事後分布の作図 -----

    # 事後分布の期待値を計算
    E_lambda = a / b
    E_sigma  = 1.0/np.sqrt(E_lambda) # (処理の効率化用)

    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$\\hat{{a}} = {a:.1f}, \\hat{{b}} = {b:.1f}$\n'
    posterior_param_lbl += f'$E[\\lambda] = \\frac{{\\hat{{a}}}}{{\\hat{{b}}}} = {E_lambda:.2f}$'

    # 事後分布を描画
    ax = axes[1, 1]
    ax.axvline(
        x=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=E_lambda, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 期待値
    ax.plot(
        lambda_vec, posterior_dens_vec, 
        color='purple', linewidth=1.0, 
        label='posterior distribution', zorder=12
    ) # 事後分布
    ax.scatter(
        x=(sigma_truth/(x_n[:n]-mu))**2, y=np.zeros(shape=n), 
        c='hotpink', alpha=0.33, s=25, 
        clip_on=False, zorder=13
    ) # 観測データ
    ax.scatter(
        x=(sigma_truth/(x-mu))**2, y=0.0, 
        c='hotpink', s=100, 
        clip_on=False, zorder=14
    ) # 観測データ

    ax.text(
        x=posterior_loc_x, y=posterior_loc_y, 
        s=posterior_param_lbl, transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('density')
    ax.set_title('Gamma distribution', loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max)  # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=posterior_dens_max) # 描画範囲を固定

    # 第2軸を描画
    ax2 = axes2x[1]
    ax2.set_xticks(
        ticks =[lambda_truth, E_lambda+1e-10], 
        labels=['$\\lambda_{{truth}}$', '$E[\\lambda]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)

    ##### 予測分布の作図 -----

    # 予測分布の期待値を計算
    E_x = mu_s

    # 予測分布のラベルを作成
    predict_param_lbl  = f'$\\mu_{{s}} = {mu_s:.1f}, \\hat{{\\lambda}}_{{s}} = {lambda_s:.3f}, \\hat{{\\nu}}_{{s}} = {nu_s:.1f}$\n'
    predict_param_lbl += f'$E[x] = \\hat{{\\mu}}_{{s}} = {E_x:.2f}$'

    # 予測分布を描画
    ax = axes[2, 0]
    ax.hlines(
        y=norm.pdf(x=mu, loc=mu, scale=sigma_truth), xmin=mu, xmax=x_max, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=mu, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 既知のパラメータ
    ax.hlines(
        y=t.pdf(x=E_x, df=nu_s, loc=E_x, scale=E_sigma), xmin=E_x, xmax=x_max, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 期待値
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
        x=x_n[:n], y=np.zeros(shape=n), 
        c='hotpink', alpha=0.33, s=25, 
        clip_on=False, zorder=14
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, 
        c='hotpink', s=100, 
        clip_on=False, zorder=15
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
        ticks =[mu, E_x+1e-10], 
        labels=['$\\mu$', '$E[x]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)

    ##### 軸変換の作図：(λ to p(λ)) -----

    # (警告文の回避用)
    tmp_lambda_vec = lambda_vec[lambda_vec > 0.0]

    # 変換曲線を描画
    ax = axes[2, 1]
    ax.plot(
        tmp_lambda_vec, norm.pdf(x=np.tile(mu, reps=len(tmp_lambda_vec)), loc=mu, scale=1.0/np.sqrt(tmp_lambda_vec)), 
        color='black', linewidth=1.0, linestyle='--', 
        label='$N(x = \\mu \\mid \\mu, \\lambda^{-1})$', zorder=10
    ) # 変換曲線:ガウス分布
    ax.plot(
        tmp_lambda_vec, t.pdf(x=np.tile(mu_s, reps=len(tmp_lambda_vec)), df=nu_s, loc=mu_s, scale=1.0/np.sqrt(tmp_lambda_vec)), 
        color='black', linewidth=1.0, 
        label='$St(x = \\mu_s \\mid \\mu_s, \\lambda, \\hat{\\nu}_s)$', zorder=11
    ) # 変換曲線:t分布
    ax.vlines(
        x=lambda_truth, ymin=norm.pdf(x=mu, loc=mu, scale=sigma_truth), ymax=lambda_max, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.hlines(
        y=norm.pdf(x=mu, loc=mu, scale=sigma_truth), xmin=lambda_min, xmax=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.vlines(
        x=E_lambda, ymin=t.pdf(x=mu_s, df=nu_s, loc=mu_s, scale=E_sigma), ymax=lambda_max, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値
    ax.hlines(
        y=t.pdf(x=mu_s, df=nu_s, loc=mu_s, scale=E_sigma), xmin=lambda_min, xmax=E_lambda, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値

    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('density')
    ax.legend(loc='lower right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=predict_dens_max)  # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../figure/gaussian_model/parameter_updates_precision/observation.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
)


# %%


