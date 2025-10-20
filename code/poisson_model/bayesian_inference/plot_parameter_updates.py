
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

### 生成分布の設定 -----

# 真のパラメータを指定
lambda_truth = 4.0


# %%

### 観測データの生成 -----

# データ数を指定
N = 100

# ポアソンモデルのデータを生成
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

# 事前分布のパラメータを指定
a = 1.0
b = 1.0


# 予測分布のパラメータを計算:式(3.44)
r = a
p = 1.0 / (b + 1.0)

# 初期値を記録
trace_a_lt = [a]
trace_b_lt = [b]
trace_r_lt = [r]
trace_p_lt = [p]

# パラメータを更新
for n in range(N):

    # 観測データを取得
    x = x_n[n]
    
    # 事後分布のパラメータを更新:式(3.38)
    a += x
    b += 1.0
    
    # 予測分布のパラメータを更新:式(3.44)
    r = a
    p = 1.0 / (b + 1.0)
    #a += x
    #p = 1.0 / (1.0/p + 1.0)
    
    # 更新値を記録
    trace_a_lt.append(a)
    trace_b_lt.append(b)
    trace_r_lt.append(r)
    trace_p_lt.append(p)

    # 動作確認
    print(f'{n+1} / {N}')


# %%

#### 一括更新の場合 -----

# 事前分布のパラメータを指定
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
trace_r_lt = np.hstack(
    [a, np.cumsum(x_n) + a]
).tolist()
trace_p_lt = (
    1.0 / (np.arange(N+1) + b + 1.0)
).tolist()


# %%

### 推移の作図 -----

#### 事後分布の作図 -----

# 事後分布の確率密度を計算:式(2.56)
trace_posterior_lt = [
    gamma.pdf(x=lambda_vec, a=trace_a_lt[i], scale=1.0/trace_b_lt[i]) for i in range(N+1)
]


# 確率密度軸の範囲を設定
u = 0.05
dens_max = np.max(trace_posterior_lt)
dens_max = np.ceil(dens_max /u)*u # u単位で切り上げ

# 図を初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white')
fig.suptitle('Gamma distribution', fontsize=20)

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
    a = trace_a_lt[i] # 形状パラメータ
    b = trace_b_lt[i] # 尺度パラメータ
    posterior_dens_vec = trace_posterior_lt[i] # 確率密度
    
    # ラベル用の文字列を作成
    posterior_param_lbl  = f'$N = {n}, '
    posterior_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.2f}, '
    posterior_param_lbl += f'\hat{{a}} == {a:.1f}, \hat{{b}} == {b:.1f}$'

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
    plt.scatter(
        x=x, y=0.0, 
        c='pink', s=100, 
        label='observation data', clip_on=False, zorder=12
    ) # 観測データ
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('density')
    ax.set_title(posterior_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid(zorder=0)
    ax.set_xlim(xmin=lambda_min, xmax=lambda_max) # 描画範囲を固定
    ax.set_ylim(ymin=0.0, ymax=dens_max) # 描画範囲を固定

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N+1, interval=100
)

# 動画を書出
anim.save(
    filename='../../figure/poisson/parameter_updates/posterior.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
)


# %%

#### 予測分布の作図 -----

# 生成分布の確率を計算:式(2.37)
model_prob_vec = poisson.pmf(k=x_vec, mu=lambda_truth)

# 予測分布の確率を計算:式(3.43)
trace_predict_lt = [
    nbinom.pmf(k=x_vec, n=trace_r_lt[i], p=1.0-trace_p_lt[i]) for i in range(N+1)
]


# 確率軸の範囲を設定
u = 0.05
prob_max = np.max(trace_predict_lt)
prob_max = np.ceil(prob_max /u)*u # u単位で切り上げ

# 図を初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white')
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
    p = trace_p_lt[i] # 失敗確率パラメータ
    predict_prob_vec = trace_predict_lt[i] # 確率
    
    # ラベル用の文字列を作成
    predict_param_lbl  = f'$N = {n}, '
    predict_param_lbl += f'\\lambda_{{truth}} = {lambda_truth:.2f}, '
    predict_param_lbl += f'\hat{{r}} == {r:.1f}, \hat{{p}} == {p:.3f}$'

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
    plt.scatter(
        x=x, y=0.0, 
        c='pink', s=100, 
        label='observation data', clip_on=False, zorder=12
    ) # 観測データ
    ax.set_xticks(ticks=x_vec)
    ax.set_xlabel('$x$')
    ax.set_ylabel('probability')
    ax.set_title(predict_param_lbl, loc='left')
    ax.legend(prop={'size': 8})
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
    filename='../../figure/poisson/parameter_updates/predict.mp4', 
    progress_callback=lambda i, n: print(f'frame: {i+1} / {n}')
)


#%%


