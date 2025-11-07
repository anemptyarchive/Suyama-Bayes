
# ポアソンモデル ----------------------------------------------------------------

# chapter 3.2.3 
# ベイズ推論
# 学習推移の可視化


#%%

# ライブラリの読込 ---------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import poisson, gamma, nbinom
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%%

# ベイズ推論の可視化 -------------------------------------------------------------

### 生成分布(ポアソン分布)の設定 -----

# 真のパラメータを指定
lambda_truth = 4.0


# %%

### 観測データの生成 -----

# シードを設定(ノートとの対応用)
#np.random.seed(86)

# データ数(試行回数)を指定
N = 100

# 観測データを生成
x_n = np.random.poisson(lam=lambda_truth, size=N)


# %%

### 変数の設定 -----

# x軸の範囲を設定
u = 5.0
x_min = 0.0
x_max = lambda_truth # 基準値を指定
x_max *= 3.0 # 倍率を指定
x_max = max(x_max, x_n.max()) # サンプルと比較
x_max = np.ceil(x_max /u)*u # u単位で切り上げ
print('x size:', x_min, x_max)

# x軸の値を作成
x_vec = np.arange(start=x_min, stop=x_max+1, step=1)


# λ軸の範囲を設定
lambda_min = x_min
lambda_max = x_max
print('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec = np.linspace(start=lambda_min, stop=lambda_max, num=1001)


# %%

### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
a = 1.0
b = 1.0

# 予測分布のパラメータを計算:式(3.44)
r = a
q = 1.0 / (1.0 + b)
p = 1.0 - q

# 初期値を記録
trace_a_lt = [a]
trace_b_lt = [b]
trace_r_lt = [r]
trace_p_lt = [p]

# ベイズ推論による更新
for n in range(N):

    # 観測データを取得
    x = x_n[n]
    
    # 事後分布のパラメータを更新:式(3.38)
    a += x
    b += 1.0
    
    # 予測分布のパラメータを更新:式(3.44)
    r = a
    q = 1.0 / (1.0 + b)
    p = b / (1.0 + b)
    #r += x
    #q = 1.0 / (1.0 + 1.0/q)
    #p = 1.0 - q
    
    # 更新値を記録
    trace_a_lt.append(a)
    trace_b_lt.append(b)
    trace_r_lt.append(r)
    trace_p_lt.append(p)

    # 動作確認
    print(f'{n+1} / {N}')


# %%

#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
a = 1.0
b = 1.0

# 事後分布のパラメータを計算:式(3.38)
trace_a_lt = np.hstack(
    [a, np.cumsum(x_n) + a]
).tolist()
trace_b_lt = (
    np.arange(N+1) + b
).tolist()

# 予測分布のパラメータを計算:式(3.44')
trace_r_lt = trace_a_lt.copy()
trace_q_lt = (
    1.0 / (1.0 + np.arange(N+1) + b)
).tolist()
trace_p_lt = (
    (np.arange(N+1) + b) / (1.0 + np.arange(N+1) + b)
).tolist()


# %%

### 推移の作図 -----

#### 事後分布の作図 -----

# 事後分布の確率密度を計算
anim_posterior_lt = [
    gamma.pdf(x=lambda_vec, a=trace_a_lt[i], scale=1.0/trace_b_lt[i]) for i in range(N+1)
]


# 確率密度軸の範囲を設定
u = 0.05
dens_max = np.max(anim_posterior_lt)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ
print(dens_max)

# 図を初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
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
    posterior_param_lbl += f'\hat{{a}} = {a:.1f}, \hat{{b}} = {b:.1f}$'

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
        x=x, y=0.0, 
        c='hotpink', s=100, 
        label='observation data', clip_on=False, zorder=12
    ) # 観測データ
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('density')
    ax.set_title(posterior_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # 描画範囲を固定
    ax.set_ylim(ymin=0.0, ymax=dens_max) # 描画範囲を固定
    
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
    filename='../../figure/poisson_model/parameter_updates/posterior.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
)


# %%

#### 予測分布の作図 -----

# 生成分布の確率を計算
model_prob_vec = poisson.pmf(k=x_vec, mu=lambda_truth)

# 予測分布の確率を計算
anim_predict_lt = [
    nbinom.pmf(k=x_vec, n=trace_r_lt[i], p=trace_p_lt[i]) for i in range(N+1)
]


# 確率軸の範囲を設定
u = 0.05
prob_max = np.max(anim_predict_lt)
prob_max = np.ceil(prob_max /u)*u # u単位で切り上げ
print(prob_max)

# 図を初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Negative Binomial distribution', fontsize=20)

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
    r = trace_r_lt[i] # 成功回数パラメータ
    p = trace_p_lt[i] # 成功確率パラメータ
    predict_prob_vec = anim_predict_lt[i] # 確率
    
    # 予測分布のラベルを作成
    predict_param_lbl  = f'$N = {n}, '
    predict_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.2f}, '
    predict_param_lbl += f'\hat{{r}} = {r:.1f}, \hat{{p}} = {p:.5f}$'

    # 予測分布を描画
    ax.bar(
        x=x_vec, height=model_prob_vec, 
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--', 
        label='true model', zorder=10
    ) # 真の分布
    ax.bar(
        x=x_vec, height=predict_prob_vec, 
        color='purple', alpha=0.5, 
        label='predict distribution', zorder=11
    ) # 予測分布
    ax.scatter(
        x=x, y=0.0, 
        c='hotpink', s=100, 
        label='observation data', clip_on=False, zorder=12
    ) # 観測データ
    ax.set_xticks(ticks=x_vec)
    ax.set_xlabel('$x$')
    ax.set_ylabel('probability')
    ax.set_title(predict_param_lbl, loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # 描画範囲を固定
    ax.set_ylim(ymin=0.0, ymax=prob_max) # 描画範囲を固定

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../figure/poisson_model/parameter_updates/predict.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
)


# %%

#### 観測データと分布の関係 -----

# ラベル位置を設定
loc_x      = x_min - 0.5
loc_margin = 0.96

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, 
    figsize=(9, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Bayesian inference', fontsize=20)
axes2 = [ax.twiny() for ax in axes] # 第2軸の設定用
ax2y = axes2[0].twinx() # 第2軸の設定用

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2.cla() for ax2 in axes2]
    ax2y.cla()
    
    # 値を取得
    n = i # データ番号
    x = x_n[i-1] if n > 0 else np.nan # 観測値
    a = trace_a_lt[i] # 形状パラメータ
    b = trace_b_lt[i] # 尺度パラメータ
    r = trace_r_lt[i] # 成功回数パラメータ
    p = trace_p_lt[i] # 成功確率パラメータ
    posterior_dens_vec = anim_posterior_lt[i] # 確率密度
    predict_prob_vec   = anim_predict_lt[i] # 確率
    
    ##### 観測データの作図 -----

    # 生成分布の期待値を計算
    E_x = lambda_truth

    # 観測データの標本平均を計算
    bar_x = np.mean(x_n[:n])
    
    # 観測データを集計
    obs_freq_vec    = np.array([np.sum(x_n[:n] == x) for x in x_vec]) # 度数
    obs_relfreq_vec = obs_freq_vec / n                                # 相対度数

    # 生成分布のラベルを作成
    model_param_lbl  = f'$N = {n}, '
    model_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.2f}, '
    model_param_lbl += f'E[x] = \\lambda_{{truth}} = {E_x:.2f}, '
    model_param_lbl += f'\\bar{{x}} = {bar_x:.2f}$'

    # 観測データを描画
    ax = axes[0]
    ax.axvline(
        x=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=bar_x, 
        color='hotpink', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 標本平均
    ax.bar(
        x=x_vec, height=model_prob_vec, 
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--', 
        label='true model', zorder=12
    ) # 真の分布
    ax.bar(
        x=x_vec, height=obs_relfreq_vec, 
        color='hotpink', alpha=0.5, 
        label='observation data', zorder=13
    ) # 観測データ
    ax.scatter(
        x=x, y=0.0, 
        c='hotpink', s=100, 
        clip_on=False, zorder=14
    ) # 観測データ
    ax.text(
        x=loc_x*loc_margin, y=prob_max*loc_margin, 
        s=model_param_lbl, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xticks(ticks=x_vec)
    ax.set_xlabel('$x$')
    ax.set_ylabel('probability')
    ax.set_title('Poisson distribution', loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=prob_max) # (目盛の共通化用)

    # 第2軸を描画
    ax2x = axes2[0]
    ax2x.set_xticks(
        ticks =[E_x, bar_x+1e-10], 
        labels=['$E[x]$', '$\\bar{x}$']
    ) # パラメータラベル
    ax2x.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # (目盛の共通化用)
    
    freq_max  = prob_max * n if n > 0 else 1.0
    prob_vals = ax.get_yticks() # 確率軸目盛を取得
    freq_vals = prob_vals * n   # 度数軸目盛に変換

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

    # 事後分布のラベルを作成
    posterior_param_lbl  = f'$\hat{{a}} = {a:.1f}, \hat{{b}} = {b:.1f}, '
    posterior_param_lbl += f'E[\\lambda] = \\frac{{\hat{{a}}}}{{\hat{{b}}}} = {E_lambda:.2f}$'

    # 事後分布を描画
    ax = axes[1]
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
        x=x, y=0.0, 
        c='hotpink', s=100, 
        clip_on=False, zorder=13
    ) # 観測データ
    ax.text(
        x=loc_x*loc_margin, y=dens_max*loc_margin, 
        s=posterior_param_lbl, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('density')
    ax.set_title('Gamma distribution', loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=dens_max) # 描画範囲を固定
    
    # 第2軸を描画
    ax2 = axes2[1]
    ax2.set_xticks(
        ticks =[lambda_truth, E_lambda+1e-10], 
        labels=['$\\lambda_{{truth}}$', '$E[\\lambda]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # (目盛の共通化用)

    ##### 予測分布の作図 -----
  
    # 予測分布の期待値を計算
    E_x = r * (1.0-p) / p
    
    # 予測分布のラベルを作成
    predict_param_lbl  = f'$\hat{{r}} = {r:.1f}, \hat{{p}} = {p:.5f}, '
    predict_param_lbl += f'E[x] = \\frac{{\hat{{r}} (1-\hat{{p}})}}{{\hat{{p}}}} = {E_x:.2f}$'

    # 予測分布を描画
    ax = axes[2]
    ax.axvline(
        x=lambda_truth, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=10
    ) # 真のパラメータ
    ax.axvline(
        x=E_x, 
        color='purple', linewidth=1.0, linestyle='--', 
        zorder=11
    ) # 期待値
    ax.bar(
        x=x_vec, height=model_prob_vec, 
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--', 
        label='true model', zorder=12
    ) # 真の分布
    ax.bar(
        x=x_vec, height=predict_prob_vec, 
        color='purple', alpha=0.5, 
        label='predict distribution', zorder=13
    ) # 予測分布
    ax.scatter(
        x=x, y=0.0, 
        c='hotpink', s=100, 
        clip_on=False, zorder=14
    ) # 観測データ
    ax.text(
        x=loc_x*loc_margin, y=prob_max*loc_margin, 
        s=predict_param_lbl, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5), 
        size = 10, 
        zorder=100
    ) # パラメータラベル
    ax.set_xticks(ticks=x_vec)
    ax.set_xlabel('$x$')
    ax.set_ylabel('probability')
    ax.set_title('Negative Binomial distribution', loc='left') # パラメータラベル
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # (目盛の共通化用)
    ax.set_ylim(ymin=0.0, ymax=prob_max) # 描画範囲を固定

    # 第2軸を描画
    ax2 = axes2[2]
    ax2.set_xticks(
        ticks =[lambda_truth, E_x+1e-10], 
        labels=['$\\lambda_{{truth}}$', '$E[x]$']
    ) # パラメータラベル
    ax2.set_xlim(xmin=x_min-0.5, xmax=x_max+0.5) # (目盛の共通化用)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../figure/poisson_model/parameter_updates/observation.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
)


#%%


