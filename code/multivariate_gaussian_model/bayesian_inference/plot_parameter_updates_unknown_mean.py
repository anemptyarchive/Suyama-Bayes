
# 多次元ガウスモデル -------------------------------------------------------------

# chapter 3.4.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


# %%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# %%

# ベイズ推論の可視化 -------------------------------------------------------------

### 生成分布の設定 -----

# 次元数を設定
D = 2

# 真のパラメータを指定
mu_truth_d = np.array(
    [25.0, 50.0]
)

# 既知のパラメータを指定
sigma2_dd = np.array(
    [[900.0, -100.0], 
     [-100.0, 400.0]]
)

# 精度行列に変換
lambda_dd = np.linalg.inv(sigma2_dd)


# %%

### 観測データの生成 -----

# シードを設定:(ノートとの対応用)
np.random.seed(86)

# データ数(試行回数)を指定
N = 300

# 観測データを生成
x_nd = np.random.multivariate_normal(mean=mu_truth_d, cov=sigma2_dd, size=N)


# %%

### 変数の設定 -----

# x軸の範囲を設定
k = 2.0
u = 5.0
x_0_size  = np.sqrt(sigma2_dd[0, 0]) # 標準偏差
x_1_size  = np.sqrt(sigma2_dd[1, 1]) # 標準偏差
x_0_size *= k # 定数倍
x_1_size *= k # 定数倍
#x_0_size  = max(x_0_size, *np.abs(x_nd[:, 0]-mu_truth_d[0])) # サンプルと比較
#x_1_size  = max(x_1_size, *np.abs(x_nd[:, 1]-mu_truth_d[1])) # サンプルと比較
x_0_size  = np.ceil(x_0_size /u)*u # u単位で切り上げ
x_1_size  = np.ceil(x_1_size /u)*u # u単位で切り上げ
x_0_min   = mu_truth_d[0] - x_0_size
x_0_max   = mu_truth_d[0] + x_0_size
x_1_min   = mu_truth_d[1] - x_1_size
x_1_max   = mu_truth_d[1] + x_1_size
print('x1 size:', x_0_min, x_1_max)
print('x2 size:', x_1_min, x_1_max)

# x軸の値を作成
x_0_vec = np.linspace(start=x_0_min, stop=x_0_max, num=251)
x_1_vec = np.linspace(start=x_1_min, stop=x_1_max, num=251)

# 格子点を作成
x_0_grid, x_1_grid = np.meshgrid(x_0_vec, x_1_vec)

# 格子点の座標を整形
x_arr  = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)
x_dims = x_0_grid.shape


# μ軸の範囲を設定
mu_0_min, mu_0_max = x_0_min, x_0_max
mu_1_min, mu_1_max = x_1_min, x_1_max
print('μ1 size:', mu_0_min, mu_0_max)
print('μ2 size:', mu_1_min, mu_1_max)

# μ軸の値を作成
mu_0_vec = np.linspace(start=mu_0_min, stop=mu_0_max, num=251)
mu_1_vec = np.linspace(start=mu_1_min, stop=mu_1_max, num=251)

# 格子点を作成
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_vec, mu_1_vec)

# 格子点の座標を作成
mu_arr  = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dims = mu_0_grid.shape


# %%

### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
m_d          = np.tile(0.0, reps=D)
sigma2_mu_dd = np.identity(D) * 100**2
lambda_mu_dd = np.linalg.inv(sigma2_mu_dd)

# 予測分布のパラメータを計算:式(3.109, 3.110)
mu_star_d      = m_d.copy()
lambda_star_dd = np.linalg.inv(
    np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_dd)
)

# 初期値を記録
trace_m_lt           = [m_d.copy()]
trace_lambda_mu_lt   = [lambda_mu_dd.copy()]
trace_mu_star_lt     = [mu_star_d.copy()]
trace_lambda_star_lt = [lambda_star_dd.copy()]

# ベイズ推論による更新
for n in range(N):

    # 観測データを取得
    x_d = x_nd[n]

    # 事後分布のパラメータを更新:式(3.102, 3.103)
    old_lambda_mu_dd = lambda_mu_dd.copy()
    lambda_mu_dd    += lambda_dd
    m_d              = (np.linalg.inv(lambda_mu_dd) @ (lambda_dd @ x_d + old_lambda_mu_dd @ m_d))

    # 予測分布のパラメータを計算:式(3.109, 3.110)
    mu_star_d      = m_d.copy()
    lambda_star_dd = np.linalg.inv(
        np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_dd)
    )
    
    # 更新値を記録
    trace_m_lt.append(m_d.copy())
    trace_lambda_mu_lt.append(lambda_mu_dd.copy())
    trace_mu_star_lt.append(mu_star_d.copy())
    trace_lambda_star_lt.append(lambda_star_dd.copy())

    # 動作確認
    print(f'\r{n+1} / {N}', end='', flush=True)
    

# %%

#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
m_d          = np.tile(0.0, reps=D)
sigma2_mu_dd = np.identity(D) * 100**2
lambda_mu_dd = np.linalg.inv(sigma2_mu_dd)

# 事後分布のパラメータを計算:式(3.102, 3.103)
trace_lambda_mu_lt = [
    n * lambda_dd + lambda_mu_dd for n in range(N+1)
]
trace_m_lt = [
    np.linalg.inv(trace_lambda_mu_lt[n]) @ (lambda_dd @ np.sum(x_nd[:n], axis=0) + lambda_mu_dd @ m_d) for n in range(N+1)
]

# 予測分布のパラメータを計算:式(3.109', 3.110')
trace_mu_star_lt     = trace_m_lt.copy()
trace_lambda_star_lt = [
    np.linalg.inv(
        np.linalg.inv(lambda_dd) + np.linalg.inv(trace_lambda_mu_lt[n])
    ) for n in range(N+1)
]


# %%

### 分布の計算 -----

# 生成分布の確率密度を計算
model_dens_grid = multivariate_normal.pdf(
    x=x_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_dd)
).reshape(x_dims)

# 事後分布の確率密度を計算
anim_posterior_lt = [
    multivariate_normal.pdf(
        x=mu_arr, mean=trace_m_lt[i], cov=np.linalg.inv(trace_lambda_mu_lt[i])
    ).reshape(mu_dims) for i in range(N+1)
]

# 予測分布の確率密度を計算
anim_predict_lt = [
    multivariate_normal.pdf(
        x=x_arr, mean=trace_mu_star_lt[i], cov=np.linalg.inv(trace_lambda_star_lt[i])
    ).reshape(x_dims) for i in range(N+1)
]


# %%

### 推移の作図 -----

# 保存先を指定
dir_path = '../../../figure/multivariate_gaussian_model/parameter_updates_unknown_mean/'

# 拡張子を指定
file_ext = '.mp4'
#file_ext = '.gif'


# %%

#### 事後分布の作図 -----

# p(μ)軸の範囲を設定
u = 0.01
dens_min = 0.0
dens_max = np.max(anim_posterior_lt)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)

# 等高線を設定
level_num = 19
dens_vals = np.linspace(start=dens_min, stop=dens_max, num=level_num)
print(dens_vals)


# %%

## 等高線図

# 図を初期化
fig, ax = plt.subplots(
    figsize=(9, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('multivariate Gaussian distribution', fontsize=20)
ax2x = ax.twiny() # 第2軸の設定用
ax2y = ax.twinx() # 第2軸の設定用
posterior_cs = ax.contourf(
    mu_0_grid, mu_1_grid, np.zeros(shape=mu_dims), 
    cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
    alpha=0.5
) # (カラーバー表示用のダミー)
fig.colorbar(posterior_cs, ax=ax, shrink=1.0, label='density')

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
    n   = i # データ番号
    x_d = x_nd[i-1] if n > 0 else np.tile(np.nan, reps=D) # 観測値
    m_d          = trace_m_lt[i]         # 平均パラメータ
    lambda_mu_dd = trace_lambda_mu_lt[i] # 精度パラメータ
    posterior_dens_grid = anim_posterior_lt[i] # 確率密度

    # 事後分布のラベルを作成
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_mu_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_mu_dd[1, :]])
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\mu_{{truth}} = \\binom{{{mu_truth_d[0]:.2f}}}{{{mu_truth_d[1]:.2f}}}, '
    posterior_param_lbl += f'\\hat{{m}} = \\binom{{{m_d[0]:.2f}}}{{{m_d[1]:.2f}}}, '
    posterior_param_lbl += f'\\hat{{\\Lambda}}_{{\\mu}} = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$'

    # 事後分布を描画
    ax.contourf(
        mu_0_grid, mu_1_grid, posterior_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        alpha=0.5, 
        zorder=10
    ) # 事後分布
    ax.plot(
        [[mu_truth_d[0], mu_0_min], [mu_truth_d[0], mu_0_max]], 
        [[mu_1_min, mu_truth_d[1]], [mu_1_max, mu_truth_d[1]]], 
        color='red', linewidth=1.0, linestyle='--', 
        label=['true parameter', None], 
        zorder=11
    ) # 真のパラメータ
    ax.plot(
        [], [], 
        color='purple', linewidth=1.0, 
        label='posterior distribution', 
        zorder=12
    ) # (凡例表示用のダミー)
    ax.scatter(
        x=x_nd[:n, 0], y=x_nd[:n, 1], 
        c='hotpink', alpha=0.33, s=25, 
        zorder=13
    ) # 観測データ
    ax.scatter(
        x=x_d[0], y=x_d[1], 
        c='hotpink', s=100, 
        label='observation data', 
        zorder=14
    ) # 観測データ

    ax.set_xlabel('$\\mu_1$')
    ax.set_ylabel('$\\mu_2$')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=mu_0_min, xmax=mu_0_max) # (目盛の共通化用)
    ax.set_ylim(ymin=mu_1_min, ymax=mu_1_max) # (目盛の共通化用)
    
    ax2x.set_xticks(ticks=[mu_truth_d[0]], labels=['$\\mu_1^{truth}$']) # パラメータラベル
    ax2y.set_yticks(ticks=[mu_truth_d[1]], labels=['$\\mu_2^{truth}$']) # パラメータラベル
    ax2x.set_xlim(xmin=mu_0_min, xmax=mu_0_max) # (目盛の共通化用)
    ax2y.set_ylim(ymin=mu_1_min, ymax=mu_1_max) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'posterior_2d'+file_ext, 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

## 曲面図

# 図を初期化
fig, ax = plt.subplots(
    figsize=(8, 6), dpi=100, facecolor='white', 
    constrained_layout=True, subplot_kw={'projection': '3d'}
)
fig.suptitle('multivariate Gaussian distribution', fontsize=20)

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()

    # 値を取得
    n   = i # データ番号
    x_d = x_nd[i-1] if n > 0 else np.tile(np.nan, reps=D) # 観測値
    m_d          = trace_m_lt[i]         # 平均パラメータ
    lambda_mu_dd = trace_lambda_mu_lt[i] # 精度パラメータ
    posterior_dens_grid = anim_posterior_lt[i] # 確率密度

    # 事後分布のラベルを作成
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_mu_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_mu_dd[1, :]])
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\mu_{{truth}} = \\binom{{{mu_truth_d[0]:.2f}}}{{{mu_truth_d[1]:.2f}}}, '
    posterior_param_lbl += f'\\hat{{m}} = \\binom{{{m_d[0]:.2f}}}{{{m_d[1]:.2f}}}, '
    posterior_param_lbl += f'\\hat{{\\Lambda}}_{{\\mu}} = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$'

    # 事後分布を描画
    ax.plot(
        [mu_truth_d[0]]*2+[np.nan]+[mu_0_min, mu_0_max], 
        [mu_1_min, mu_1_max]+[np.nan]+[mu_truth_d[1]]*2, 
        [0.0, 0.0, np.nan, 0.0, 0.0], 
        color='red', linewidth=1.0, linestyle='--', 
        label='true parameter', 
        zorder=10
    ) # 真のパラメータ
    ax.contour(
        X=mu_0_grid, Y=mu_1_grid, Z=posterior_dens_grid, offset=0.0, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        linewidths=1.0, linestyles=':', 
        zorder=11
    ) # 事後分布
    ax.contour(
        X=mu_0_grid, Y=mu_1_grid, Z=posterior_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        linewidths=1.0, linestyles='-', 
        zorder=12
    ) # 事後分布
    ax.plot_surface(
        X=mu_0_grid, Y=mu_1_grid, Z=posterior_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, 
        alpha=0.8, 
        label='posterior distribution', 
        zorder=13
    ) # 事後分布
    tmp_bool_n = (x_nd[:n, 0] >= x_0_min) & (x_nd[:n, 0] <= x_0_max) & (x_nd[:n, 1] >= x_1_min) & (x_nd[:n, 1] <= x_1_max)
    ax.scatter(
        xs=x_nd[:n, 0][tmp_bool_n], ys=x_nd[:n, 1][tmp_bool_n], zs=0.0, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=14
    ) # 観測データ
    tmp_bool  = (x_d[0] >= x_0_min) & (x_d[0] <= x_0_max) & (x_d[1] >= x_1_min) & (x_d[1] <= x_1_max)
    tmp_val_d = x_d if tmp_bool else np.tile(np.nan, reps=D)
    ax.scatter(
        xs=tmp_val_d[0], ys=tmp_val_d[1], zs=0.0, 
        c='hotpink', s=100, 
        label='observation data', 
        zorder=15
    ) # 観測データ
    
    ax.set_xlabel('$\\mu_1$')
    ax.set_ylabel('$\\mu_2$')
    ax.set_zlabel('density')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.set_xlim(xmin=mu_0_min, xmax=mu_0_max)
    ax.set_ylim(ymin=mu_1_min, ymax=mu_1_max)
    ax.set_zlim(zmin=dens_min, zmax=dens_max) # 描画範囲を固定
    #ax.view_init(elev=30, azim=-60+i) # 表示角度

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'posterior_3d'+file_ext, 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

#### 予測分布の作図 -----

# p(x)軸の範囲を設定
u = 0.00005
dens_min = 0.0
dens_max = np.max(model_dens_grid)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)

# 等高線を設定
model_cs = plt.contour(
    x_0_grid, x_1_grid, model_dens_grid, 
    vmin=dens_min, vmax=dens_max
) # 真の分布
dens_vals = model_cs.levels
print(dens_vals)


# %%

## 等高線図

# 図を初期化
fig, ax = plt.subplots(
    figsize=(9, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('multivariate Gaussian distribution', fontsize=20)
predict_cs = ax.contourf(
    x_0_grid, x_1_grid, np.zeros(shape=x_dims), 
    cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
    alpha=0.5
) # (カラーバー表示用のダミー)
fig.colorbar(predict_cs, ax=ax, shrink=1.0, label='density')

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()

    # 値を取得
    n   = i # データ番号
    x_d = x_nd[i-1] if n > 0 else np.tile(np.nan, reps=D) # 観測値
    mu_star_d      = trace_mu_star_lt[i]     # 平均パラメータ
    lambda_star_dd = trace_lambda_star_lt[i] # 精度パラメータ
    predict_dens_grid = anim_predict_lt[i] # 確率密度

    # 予測分布のラベルを作成
    predict_param_lbl  = f'$N = {n}, '
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_dd[1, :]])
    predict_param_lbl += f'\\mu_{{truth}} = \\binom{{{mu_truth_d[0]:.2f}}}{{{mu_truth_d[1]:.2f}}}, '
    predict_param_lbl += f'\\Lambda = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}, '
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_star_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_star_dd[1, :]])
    predict_param_lbl += f'\\hat{{\\mu}}_{{*}} = \\binom{{{mu_star_d[0]:.2f}}}{{{mu_star_d[1]:.2f}}}, '
    predict_param_lbl += f'\\hat{{\\Lambda}}_{{*}} = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$'

    # 予測分布を描画
    ax.contourf(
        x_0_grid, x_1_grid, predict_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        alpha=0.5, 
        zorder=10
    ) # 予測分布
    ax.contour(
        x_0_grid, x_1_grid, model_dens_grid, 
        cmap='YlOrRd_r', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        linewidths=1.0, linestyles='--', 
        zorder=11
    ) # 真の分布
    model_dummy = plt.Line2D(
        [], [], 
        color='red', linewidth=1.0, linestyle='--', 
        label='true model'
    ) # (凡例表示用のダミー)
    predict_dummy = mpatches.Patch(
        color='purple', 
        label='predict distribution'
    ) # (凡例表示用のダミー)
    ax.scatter(
        x=x_nd[:n, 0], y=x_nd[:n, 1], 
        c='hotpink', alpha=0.33, s=25, 
        zorder=12
    ) # 観測データ
    obs_handle = ax.scatter(
        x=x_d[0], y=x_d[1], 
        c='hotpink', s=100, 
        label='observation data', 
        zorder=13
    ) # 観測データ

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(predict_param_lbl, loc='left') # パラメータラベル
    ax.legend(
        handles=[model_dummy, predict_dummy, obs_handle], 
        loc='upper right', prop={'size': 8}
    )
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_0_min, xmax=x_0_max)
    ax.set_ylim(ymin=x_1_min, ymax=x_1_max)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'predict_2d'+file_ext, 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

## 曲面図

# 図を初期化
fig, ax = plt.subplots(
    figsize=(8, 6), dpi=100, facecolor='white', 
    constrained_layout=True, subplot_kw={'projection': '3d'}
)
fig.suptitle('multivariate Gaussian distribution', fontsize=20)

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    ax.cla()
    
    # 値を取得
    n   = i # データ番号
    x_d = x_nd[i-1] if n > 0 else np.tile(np.nan, reps=D) # 観測値
    mu_star_d      = trace_mu_star_lt[i]     # 平均パラメータ
    lambda_star_dd = trace_lambda_star_lt[i] # 精度パラメータ
    predict_dens_grid = anim_predict_lt[i] # 確率密度

    # 予測分布のラベルを作成
    predict_param_lbl  = f'$N = {n}, '
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_dd[1, :]])
    predict_param_lbl += f'\\mu_{{truth}} = \\binom{{{mu_truth_d[0]:.2f}}}{{{mu_truth_d[1]:.2f}}}, '
    predict_param_lbl += f'\\Lambda = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}, '
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_star_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_star_dd[1, :]])
    predict_param_lbl += f'\\hat{{\\mu}}_{{*}} = \\binom{{{mu_star_d[0]:.2f}}}{{{mu_star_d[1]:.2f}}}, '
    predict_param_lbl += f'\\hat{{\\Lambda}}_{{*}} = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$'

    ax.plot_wireframe(
        X=x_0_grid, Y=x_1_grid, Z=model_dens_grid, 
        color='red', linewidth=0.5, linestyle='--', 
        label='true model', 
        zorder=11
    ) # 真の分布
    ax.contour(
        X=x_0_grid, Y=x_1_grid, Z=predict_dens_grid, offset=0.0, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 予測分布
    ax.contour(
        X=x_0_grid, Y=x_1_grid, Z=predict_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, 
        linewidths=1.0, linestyles='-', 
        zorder=12
    ) # 予測分布
    ax.plot_surface(
        X=x_0_grid, Y=x_1_grid, Z=predict_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=dens_max, 
        alpha=0.8, 
        label='predict distribution', 
        zorder=13
    ) # 予測分布
    tmp_bool_n = (x_nd[:n, 0] >= x_0_min) & (x_nd[:n, 0] <= x_0_max) & (x_nd[:n, 1] >= x_1_min) & (x_nd[:n, 1] <= x_1_max)
    ax.scatter(
        xs=x_nd[:n, 0][tmp_bool_n], ys=x_nd[:n, 1][tmp_bool_n], zs=0.0, 
        c='hotpink', alpha=0.33, s=25, 
        zorder=14
    ) # 観測データ
    tmp_bool  = (x_d[0] >= x_0_min) & (x_d[0] <= x_0_max) & (x_d[1] >= x_1_min) & (x_d[1] <= x_1_max)
    tmp_val_d = x_d if tmp_bool else np.tile(np.nan, reps=D)
    ax.scatter(
        xs=tmp_val_d[0], ys=tmp_val_d[1], zs=0.0, 
        c='hotpink', s=100, 
        label='observation data', 
        zorder=15
    ) # 観測データ

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('density')
    ax.set_title(predict_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.set_xlim(xmin=x_0_min, xmax=x_0_max)
    ax.set_ylim(ymin=x_1_min, ymax=x_1_max)
    ax.set_zlim(zmin=dens_min, zmax=dens_max) # 描画範囲を固定
    #ax.view_init(elev=30, azim=-60+i) # 表示角度

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'predict_3d'+file_ext, 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

#### 観測データと分布の関係 -----

# 階級数を指定
class_0_num = 31
class_1_num = 21

# 階級値・境界値を作成
bin_0_num    = class_0_num + 1 # 境界数
bin_1_num    = class_1_num + 1 # 境界数
bin_0_size   = (x_0_max - x_0_min) / (class_0_num-1) # 階級幅
bin_1_size   = (x_1_max - x_1_min) / (class_1_num-1) # 階級幅
bin_0_min    = x_0_min - 0.5*bin_0_size # 境界値の最小値
bin_0_max    = x_0_max + 0.5*bin_0_size # 境界値の最大値
bin_1_min    = x_1_min - 0.5*bin_1_size # 境界値の最小値
bin_1_max    = x_1_max + 0.5*bin_1_size # 境界値の最大値
bin_0_vec    = np.linspace(start=bin_0_min, stop=bin_0_max, num=bin_0_num) # 境界値
bin_1_vec    = np.linspace(start=bin_1_min, stop=bin_1_max, num=bin_1_num) # 境界値
center_0_vec = bin_0_vec[:-1] + 0.5*bin_0_size # 階級値
center_1_vec = bin_1_vec[:-1] + 0.5*bin_1_size # 階級値
print('bin size:', bin_0_size, bin_1_size)


# p(μ)軸の範囲を設定
u = 0.01
dens_min = 0.0
posterior_dens_max = np.max(anim_posterior_lt)
posterior_dens_max = np.ceil(posterior_dens_max /u)*u # u単位で切り上げ
print(posterior_dens_max)

# 等高線を設定
level_num = 19
posterior_dens_vals = np.linspace(start=dens_min, stop=posterior_dens_max, num=level_num)
print(posterior_dens_vals)

# p(x)軸の範囲を設定
u = 0.00005
predict_dens_max = np.max(model_dens_grid)
predict_dens_max = np.ceil(predict_dens_max /u)*u # u単位で切り上げ
print(predict_dens_max)

# 等高線を設定
model_cs = plt.contour(
    x_0_grid, x_1_grid, model_dens_grid, 
    vmin=dens_min, vmax=predict_dens_max
) # 真の分布
predict_dens_vals = model_cs.levels
print(predict_dens_vals)

# %%

# ラベル位置を設定
loc_x = 0.02
loc_y = 0.98

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=3, 
    figsize=(15, 15), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Bayesian inference', fontsize=20)
axes2x = np.array([[ax.twiny() for ax in row] for row in axes]) # 第2軸の設定用
axes2y = np.array([[ax.twinx() for ax in row] for row in axes]) # 第2軸の設定用

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes.flatten()]
    [ax.cla() for ax in axes2x.flatten()]
    [ax.cla() for ax in axes2y.flatten()]

    ##### パラメータの設定 -----

    # 値を取得
    n   = i # データ番号
    x_d = x_nd[i-1] if n > 0 else np.tile(np.nan, reps=D) # 観測値
    m_d            = trace_m_lt[i]           # 平均パラメータ
    lambda_mu_dd   = trace_lambda_mu_lt[i]   # 精度パラメータ
    mu_star_d      = trace_mu_star_lt[i]     # 平均パラメータ
    lambda_star_dd = trace_lambda_star_lt[i] # 精度パラメータ
    posterior_dens_grid = anim_posterior_lt[i] # 確率密度
    predict_dens_grid   = anim_predict_lt[i]   # 確率密度

    ##### 観測データの作図 -----

    # 生成分布の期待値を計算
    E_x_d = mu_truth_d

    # 観測データの標本平均を計算
    bar_x_d = np.mean(x_nd[:n], axis=0) if n > 0 else np.tile(np.nan, reps=D)

    # 観測データを集計
    if n > 0:
        obs_dens_grid, _, _ = np.histogram2d(
            x=x_nd[:n, 0], y=x_nd[:n, 1], 
            bins=(class_0_num, class_1_num), 
            range=[(bin_0_min, bin_0_max), (bin_1_min, bin_1_max)], 
            density=True
        )
    else:
        obs_dens_grid = np.zeros(shape=(class_0_num, class_1_num)) # (警告文の回避用)
    
    # 描画処理を定義
    def draw_model(ax, ax2x, ax2y, lbl):

        # 生成分布を描画
        ax.pcolormesh(
            center_0_vec, center_1_vec, obs_dens_grid.T, 
            shading='nearest', 
            cmap='YlOrRd_r', #vmin=dens_min, vmax=predict_dens_max, ## (Nが不十分な場合はスケールを統一しない方が意図に近い図になる)
            alpha=0.5, 
            zorder=10
        ) # 観測データ
        ax.contour(
            x_0_grid, x_1_grid, model_dens_grid, 
            cmap='YlOrRd_r', vmin=dens_min, vmax=predict_dens_max, levels=predict_dens_vals, 
            linewidths=1.0, linestyles='--', 
            zorder=11
        ) # 生成分布
        ax.plot(
            [[mu_truth_d[0], x_0_min], [mu_truth_d[0], x_0_max]], 
            [[x_1_min, mu_truth_d[1]], [x_1_max, mu_truth_d[1]]], 
            color='red', linewidth=1.0, linestyle='--', 
            zorder=12
        ) # 真のパラメータ
        ax.plot(
            [[bar_x_d[0], x_0_min], [bar_x_d[0], x_0_max]], 
            [[x_1_min, bar_x_d[1]], [x_1_max, bar_x_d[1]]], 
            color='hotpink', linewidth=1.0, linestyle='--', 
            zorder=13
        ) # 標本平均
        ax.scatter(
            x=x_nd[:n, 0], y=x_nd[:n, 1], 
            c='hotpink', alpha=0.33, s=25, 
            zorder=14
        ) # 観測データ
        ax.scatter(
            x=x_d[0], y=x_d[1], 
            c='hotpink', s=100, 
            zorder=15
        ) # 観測データ
        model_dummy = plt.Line2D(
            [], [], 
            color='red', linewidth=1.0, linestyle='--', 
            label='true model'
        ) # (凡例表示用のダミー)
        obs_dummy = mpatches.Patch(
            color='hotpink', 
            label='observation data'
        ) # (凡例表示用のダミー)

        ax.text(
            x=loc_x, y=loc_y, 
            s=lbl, transform=ax.transAxes, ha='left', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
            size = 10, 
            zorder=100
        ) # パラメータラベル
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('multivariate Gaussian distribution')
        ax.legend(handles=[model_dummy, obs_dummy], loc='upper right', prop={'size': 8})
        ax.grid(zorder=0)
        ax.set_xlim(xmin=x_0_min, xmax=x_0_max) # (目盛の共通化用)
        ax.set_ylim(ymin=x_1_min, ymax=x_1_max) # (目盛の共通化用)

        ax2x.set_xticks(
            ticks =[E_x_d[0], bar_x_d[0]+1e-10], 
            labels=['$E[x_1]$', '$\\bar{x_1}$']
        ) # パラメータラベル
        ax2y.set_yticks(
            ticks =[E_x_d[1], bar_x_d[1]+1e-10], 
            labels=['$E[x_2]$', '$\\bar{x_2}$']
        ) # パラメータラベル
        ax2x.set_xlim(xmin=x_0_min, xmax=x_0_max) # (目盛の共通化用)
        ax2y.set_ylim(ymin=x_1_min, ymax=x_1_max) # (目盛の共通化用)

    # 生成分布のラベルを作成
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_dd[1, :]])
    obs_param_lbl  = f'$\\mu_{{truth}} = \\binom{{{mu_truth_d[0]:.2f}}}{{{mu_truth_d[1]:.2f}}}, '
    obs_param_lbl += f'\\Lambda = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$\n'
    obs_param_lbl += f'$E[x] = \\mu_{{truth}} = \\binom{{{E_x_d[0]:.2f}}}{{{E_x_d[1]:.2f}}}$\n'
    obs_param_lbl += f'$\\bar{{x}} = \\binom{{{bar_x_d[0]:.2f}}}{{{bar_x_d[1]:.2f}}}$'

    # 生成分布を描画
    draw_model(axes[0, 2], axes2x[0, 2], axes2y[0, 2], f'$N = {n}$')
    draw_model(axes[2, 0], axes2x[2, 0], axes2y[2, 0], obs_param_lbl)
    
    ##### 事後分布の作図 -----

    # 事後分布の期待値を計算
    E_mu_d = m_d

    # 描画処理を定義
    def draw_posterior(ax, ax2x, ax2y, lbl):

        # 事後分布を描画
        ax.contourf(
            mu_0_grid, mu_1_grid, posterior_dens_grid, 
            cmap='viridis', vmin=dens_min, vmax=posterior_dens_max, levels=posterior_dens_vals, 
            alpha=0.5, 
            zorder=10
        ) # 事後分布
        ax.plot(
            [[mu_truth_d[0], mu_0_min], [mu_truth_d[0], mu_0_max]], 
            [[mu_1_min, mu_truth_d[1]], [mu_1_max, mu_truth_d[1]]], 
            color='red', linewidth=1.0, linestyle='--', 
            label=['true parameter', None], 
            zorder=11
        ) # 真のパラメータ
        ax.plot(
            [[E_mu_d[0], mu_0_min], [E_mu_d[0], mu_0_max]], 
            [[mu_1_min, E_mu_d[1]], [mu_1_max, E_mu_d[1]]], 
            color='purple', linewidth=1.0, linestyle='--', 
            zorder=12
        ) # 期待値
        ax.scatter(
            x=x_nd[:n, 0], y=x_nd[:n, 1], 
            c='hotpink', alpha=0.33, s=25, 
            zorder=13
        ) # 観測データ
        ax.scatter(
            x=x_d[0], y=x_d[1], 
            c='hotpink', s=100, 
            zorder=14
        ) # 観測データ
        ax.plot(
            [], [], 
            color='purple', linewidth=1.0, 
            label='prior distribution'
        ) # (凡例表示用のダミー)

        ax.text(
            x=loc_x, y=loc_y, 
            s=lbl, transform=ax.transAxes, ha='left', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
            size = 10, 
            zorder=100
        ) # パラメータラベル
        ax.set_xlabel('$\\mu_1$')
        ax.set_ylabel('$\\mu_2$')
        ax.set_title('multivariate Gaussian distribution')
        ax.legend(loc='upper right', prop={'size': 8})
        ax.grid(zorder=0)
        ax.set_xlim(xmin=mu_0_min, xmax=mu_0_max) # (目盛の共通化用)
        ax.set_ylim(ymin=mu_1_min, ymax=mu_1_max) # (目盛の共通化用)

        ax2x.set_xticks(
            ticks =[mu_truth_d[0], E_mu_d[0]+1e-10], 
            labels=['$\\mu_1^{truth}$', '$E[\\mu_1]$']
        ) # パラメータラベル
        ax2y.set_yticks(
            ticks =[mu_truth_d[1], E_mu_d[1]+1e-10], 
            labels=['$\\mu_2^{truth}$', '$E[\\mu_2]$']
        ) # パラメータラベル
        ax2x.set_xlim(xmin=mu_0_min, xmax=mu_0_max) # (目盛の共通化用)
        ax2y.set_ylim(ymin=mu_1_min, ymax=mu_1_max) # (目盛の共通化用)

    # 事後分布のラベルを作成
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_mu_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_mu_dd[1, :]])
    posterior_param_lbl  = f'$\\hat{{m}} = \\binom{{{m_d[0]:.2f}}}{{{m_d[1]:.2f}}}, '
    posterior_param_lbl += f'\\hat{{\\Lambda}}_{{\\mu}} = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$\n'
    posterior_param_lbl += f'$E[\\mu] = \\hat{{m}} = \\binom{{{E_mu_d[0]:.2f}}}{{{E_mu_d[1]:.2f}}}$'

    # 事前分布を描画
    draw_posterior(axes[1, 2], axes2x[1, 2], axes2y[1, 2], None)
    draw_posterior(axes[2, 1], axes2x[2, 1], axes2y[2, 1], posterior_param_lbl)

    ##### 予測分布の作図 -----

    # 予測分布の期待値を計算
    E_x_star_d = mu_star_d

    # 予測分布のラベルを作成
    tmp_lmd_0_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_star_dd[0, :]])
    tmp_lmd_1_str = ', '.join([f'{lmd:.5f}' for lmd in lambda_star_dd[1, :]])
    predict_param_lbl  = f'$\\hat{{\\mu}}_{{*}} = \\binom{{{mu_star_d[0]:.2f}}}{{{mu_star_d[1]:.2f}}}, '
    predict_param_lbl += f'\\hat{{\\Lambda}}_{{*}} = \\binom{{{tmp_lmd_0_str}}}{{{tmp_lmd_1_str}}}$\n'
    predict_param_lbl += f'$E[x_{{*}}] = \\hat{{\\mu}}_{{*}} = \\binom{{{E_x_star_d[0]:.2f}}}{{{E_x_star_d[1]:.2f}}}$'
    
    # 予測分布を描画
    ax   = axes[2, 2]
    ax2x = axes2x[2, 2]
    ax2y = axes2y[2, 2]
    ax.contourf(
        x_0_grid, x_1_grid, predict_dens_grid, 
        cmap='viridis', vmin=dens_min, vmax=predict_dens_max, levels=predict_dens_vals, 
        alpha=0.5, 
        zorder=10
    ) # 予測分布
    ax.contour(
        x_0_grid, x_1_grid, model_dens_grid, 
        cmap='YlOrRd_r', vmin=dens_min, vmax=predict_dens_max, levels=predict_dens_vals, 
        linewidths=1.0, linestyles='--', 
        zorder=11
    ) # 真の分布
    ax.plot(
        [[mu_truth_d[0], x_0_min], [mu_truth_d[0], x_0_max]], 
        [[x_1_min, mu_truth_d[1]], [x_1_max, mu_truth_d[1]]], 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=12
    ) # 真のパラメータ
    ax.plot(
        [[E_x_star_d[0], x_0_min], [E_x_star_d[0], x_0_max]], 
        [[x_1_min, E_x_star_d[1]], [x_1_max, E_x_star_d[1]]], 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=13
    ) # 期待値
    ax.scatter(
        x=x_nd[:n, 0], y=x_nd[:n, 1], 
        c='hotpink', alpha=0.33, s=25, 
        zorder=14
    ) # 観測データ
    ax.scatter(
        x=x_d[0], y=x_d[1], 
        c='hotpink', s=100, 
        zorder=15
    ) # 観測データ
    model_dummy = plt.Line2D(
        [], [], 
        color='red', linewidth=1.0, linestyle='--', 
        label='true model'
    ) # (凡例表示用のダミー)
    predict_dummy = mpatches.Patch(
        color='purple', 
        label='predict distribution'
    ) # (凡例表示用のダミー)

    ax.text(
        x=loc_x, y=loc_y, 
        s=predict_param_lbl, transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('multivariate Gaussian distribution')
    ax.legend(handles=[model_dummy, predict_dummy], prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_0_min, xmax=x_0_max) # (目盛の共通化用)
    ax.set_ylim(ymin=x_1_min, ymax=x_1_max) # (目盛の共通化用)

    ax2x.set_xticks(
        ticks =[mu_truth_d[0], E_x_star_d[0]+1e-10], 
        labels=['$\\mu_1^{truth}$', '$E[x_1^{*}]$']
    ) # パラメータラベル
    ax2y.set_yticks(
        ticks =[mu_truth_d[1], E_x_star_d[1]+1e-10], 
        labels=['$\\mu_2^{truth}$', '$E[x_2^{*}]$']
    ) # パラメータラベル
    ax2x.set_xlim(xmin=x_0_min, xmax=x_0_max) # (目盛の共通化用)
    ax2y.set_ylim(ymin=x_1_min, ymax=x_1_max) # (目盛の共通化用)

    ##### 非表示の設定 -----

    # 不要な領域を非表示化
    [ax.axis('off') for ax in axes[:2, :2].flatten()]
    [ax.axis('off') for ax in axes2x[:2, :2].flatten()]
    [ax.axis('off') for ax in axes2y[:2, :2].flatten()]

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'observation'+file_ext, 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


