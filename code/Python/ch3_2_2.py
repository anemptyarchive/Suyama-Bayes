# 3.2.2 カテゴリ分布の学習と予測

#%%

# 3.2.2項で利用するライブラリ
import numpy as np
from scipy.special import gammaln # 対数ガンマ関数
from scipy.stats import dirichlet # ディリクレ分布
import matplotlib.pyplot as plt

#%%

## 尤度(カテゴリ分布)の設定

# 次元数の設定:(固定)
K = 3

# 真のパラメータを指定
pi_truth_k = np.array([0.3, 0.5, 0.2])


# 作図用の次元番号を作成
k_line = np.arange(1, K + 1)

# 尤度の確率を計算:式(2.29)
model_prob = pi_truth_k.copy()

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.bar(x=k_line, height=pi_truth_k, label='true model') # 真の分布
plt.xlabel('k')
plt.ylabel('prob')
plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$\pi=(' + ', '.join([str(pi) for pi in  pi_truth_k]) + ')$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.ylim(0.0, 1.0) # y軸の表示範囲
plt.show()

#%%

## 観測データの生成

# データ数を指定
N = 50

# (観測)データを生成
s_nk = np.random.multinomial(n=1, pvals=pi_truth_k, size=N)

#%%

# 観測データのヒストグラムを作図
plt.figure(figsize=(12, 9))
plt.bar(x=k_line, height=pi_truth_k, 
        color='white', edgecolor='red', linestyle='--', label='true model') # 真の分布
plt.bar(x=k_line, height=np.sum(s_nk, axis=0) / N, 
        alpha=0.6, label='data') # 観測データ:(相対度数)
#plt.bar(x=k_line, height=np.sum(s_nk, axis=0), label='data') # 観測データ:(度数)
plt.xlabel('k')
plt.ylabel('density')
plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \pi=(' + ', '.join([str(pi) for pi in pi_truth_k]) + ')$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.ylim(0.0, 1.0) # y軸の表示範囲
plt.show()

#%%

## 事前分布(ディリクレ分布)の設定

# 事前分布のパラメータを指定
alpha_k = np.repeat(1.0, K)


# 作図用のpiの値を作成
pi_line = np.arange(0.0, 1.001, 0.02)

# 格子状の点を作成
pi_0_grid, pi_1_grid, pi_2_grid = np.meshgrid(pi_line, pi_line, pi_line)

# 計算用の配列を作成
pi_point_arr = np.stack([pi_0_grid.flatten(), pi_1_grid.flatten(), pi_2_grid.flatten()], axis=1) # 結合
pi_point_arr = pi_point_arr[1:, :] # (0, 0, 0)の行を除去
pi_point_arr /= np.sum(pi_point_arr, axis=1, keepdims=True) # 正規化
pi_point_arr = np.unique(pi_point_arr, axis=0) # 重複を除去
print(pi_point_arr.shape)

# 三角座標に変換
tri_x = pi_point_arr[:, 1] + 0.5 * pi_point_arr[:, 2]
tri_y = np.sqrt(3.0) * 0.5 * pi_point_arr[:, 2]


# 真のパラメータの値を三角座標に変換
tri_x_truth = pi_truth_k[1] + 0.5 * pi_truth_k[2]
tri_y_truth = np.sqrt(3.0) * 0.5 * pi_truth_k[2]


# 事前分布の確率密度を計算:式(2.41)
ln_C_Dir = gammaln(np.sum(alpha_k)) - np.sum(gammaln(alpha_k)) # 正規化項(対数)
prior_dens = np.exp(ln_C_Dir) * np.prod(pi_point_arr**(alpha_k - 1), axis=1)
#prior_dens = dirichlet.pdf(x=pi_point_arr.T, alpha=alpha_k)

#%%

# 事前分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(tri_x, tri_y, c=prior_dens, cmap='jet', label='prior') # 事前分布
plt.colorbar() # z軸の値
plt.scatter(tri_x_truth, tri_y_truth, 
            color='red', marker='x', s=200, linewidth=3, label='true val') # 真の値
plt.xlabel('$\pi_1, \pi_2$') # x軸ラベル
plt.ylabel('$\pi_1, \pi_3$') # y軸ラベル
plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)']) # x軸目盛
plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)']) # y軸目盛
plt.suptitle('Dirichlet Distribution', fontsize=20)
plt.title('$\\alpha=(' + ', '.join([str(a) for a in alpha_k]) + ')$', loc='left')
plt.legend() # 凡例
plt.gca().set_aspect('equal') # アスペクト比
plt.show()

#%%

## 事後分布(ディリクレ分布)の計算

# 事後分布のパラメータを計算:式(3.28)
alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k


# 事後分布の確率密度を計算:式(2.41)
ln_C_Dir = gammaln(np.sum(alpha_hat_k)) - np.sum(gammaln(alpha_hat_k)) # 正規化項(対数)
#posterior_dens = np.exp(ln_C_Dir) * np.prod(pi_point_arr**(alpha_hat_k - 1), axis=1)
posterior_dens = dirichlet.pdf(x=pi_point_arr.T, alpha=alpha_hat_k)

#%%

# 事後分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(tri_x, tri_y, c=posterior_dens, cmap='jet', label='posterior') # 事後分布
plt.colorbar() # z軸の値
plt.scatter(tri_x_truth, tri_y_truth, 
            color='red', marker='x', s=200, linewidth=3, label='true val') # 真の値
plt.xlabel('$\pi_1, \pi_2$') # x軸ラベル
plt.ylabel('$\pi_1, \pi_3$') # y軸ラベル
plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)']) # x軸目盛
plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)']) # y軸目盛
plt.suptitle('Dirichlet Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{\\alpha}=(' + ', '.join([str(a) for a in alpha_hat_k]) + ')$', loc='left')
plt.legend() # 凡例
plt.gca().set_aspect('equal') # アスペクト比
plt.show()

#%%

## 予測分布()の計算

# 予測分布のパラメータを計算:式(3.31')
pi_star_hat_k = alpha_hat_k / np.sum(alpha_hat_k)
#pi_star_hat_k = (np.sum(s_nk, axis=0) + alpha_k) / np.sum(np.sum(s_nk, axis=0) + alpha_k)

#%%

# 予測分布(カテゴリ分布)を作図
plt.figure(figsize=(12, 9))
plt.bar(x=k_line, height=pi_truth_k,
        color='white', edgecolor='red', linestyle='--', label='true model') # 真の分布
plt.bar(x=k_line, height=pi_star_hat_k, 
        alpha=0.6, color='purple', label='predict') # 予測分布
plt.xlabel('k')
plt.ylabel('prob')
plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{\pi}_{*}=(' + ', '.join([str(pi) for pi in np.round(pi_star_hat_k, 2)]) + ')$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.ylim(0.0, 1.0) # y軸の表示範囲
plt.show()

#%%

### アニメーションによる推移の確認

# 3.2.2項で利用するライブラリ
import numpy as np
from scipy.stats import dirichlet # ディリクレ分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# 次元数を設定:(固定)
K = 3

# 真のパラメータを指定
pi_truth_k = np.array([0.3, 0.5, 0.2])

# 事前分布のパラメータを指定
alpha_k = np.repeat(1.0, K)

# 初期値による予測分布のパラメータを計算:式(3.31)
pi_star_k = alpha_k / np.sum(alpha_k)


# 作図用のpiの値を作成
pi_line = np.arange(0.0, 1.001, 0.025)

# 格子状の点を作成
pi_0_grid, pi_1_grid, pi_2_grid = np.meshgrid(pi_line, pi_line, pi_line)

# 計算用の配列を作成
pi_point_arr = np.stack([pi_0_grid.flatten(), pi_1_grid.flatten(), pi_2_grid.flatten()], axis=1) # 結合
pi_point_arr = pi_point_arr[1:, :] # (0, 0, 0)の行を除去
pi_point_arr /= np.sum(pi_point_arr, axis=1, keepdims=True) # 正規化
pi_point_arr = np.unique(pi_point_arr, axis=0) # 重複を除去
print(pi_point_arr.shape)

# 三角座標に変換
tri_x = pi_point_arr[:, 1] + 0.5 * pi_point_arr[:, 2]
tri_y = np.sqrt(3.0) * 0.5 * pi_point_arr[:, 2]

#%%

## 推論処理

# データ数を指定
N = 100

# 観測データの受け皿を作成
s_nk = np.empty((N, K))

# 記録用の受け皿を初期化
trace_alpha = [alpha_k]
trace_posterior = [dirichlet.pdf(x=pi_point_arr.T, alpha=alpha_k)]
trace_pi = [pi_star_k]

# ベイズ推論
for n in range(N):
    # カテゴリ分布に従うデータを生成
    s_nk[n] = np.random.multinomial(n=1, pvals=pi_truth_k, size=1).flatten()
    
    # 事後分布のパラメータを更新:式(3.28)
    alpha_k += s_nk[n]
    
    # 事後分布(ディリクレ分布)の確率密度を計算:式(2.48)
    trace_posterior.append(
        dirichlet.pdf(x=pi_point_arr.T, alpha=alpha_k)
    )
    
    # 予測分布のパラメータを更新:式(3.31)
    pi_star_k = alpha_k / np.sum(alpha_k)
    
    # n回目の結果を記録
    trace_alpha.append(alpha_k.copy())
    trace_pi.append(pi_star_k.copy())

# 観測のデータを確認
print(np.sum(s_nk, axis=0))

#%%

## 事後分布の推移をgif画像化

# 真のパラメータの値を三角座標に変換
tri_x_truth = pi_truth_k[1] + 0.5 * pi_truth_k[2]
tri_y_truth = np.sqrt(3.0) * 0.5 * pi_truth_k[2]

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n番目の観測データを三角座標に変換
    if n > 0: # 初回は除く
        tri_x_data = s_nk[n-1, 1] + 0.5 * s_nk[n-1, 2]
        tri_y_data = np.sqrt(3.0) * 0.5 * s_nk[n-1, 2]
    
    # n回目の事後分布を作図
    plt.scatter(tri_x, tri_y, c=trace_posterior[n], cmap='jet', label='posterior') # 事後分布
    plt.scatter(tri_x_truth, tri_y_truth, 
                color='red', marker='x', s=200, linewidth=3, label='true val') # 真の値
    if n > 0: # 初回は除く
        plt.scatter(x=tri_x_data, y=tri_y_data, 
                    s=200, color='purple', label='data') # 観測データ
    plt.xlabel('$\pi_1, \pi_2$') # x軸ラベル
    plt.ylabel('$\pi_1, \pi_3$') # y軸ラベル
    plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)']) # x軸目盛
    plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)']) # y軸目盛
    plt.suptitle('Dirichlet Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\\alpha}=(' + ', '.join([str(a) for a in trace_alpha[n]]) + ')$', loc='left')
    plt.legend() # 凡例
    plt.gca().set_aspect('equal') # アスペクト比

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_2_2_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# 作図用の次元番号を作成
k_line = np.arange(1, K + 1)

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # n回目の予測分布を作図
    plt.bar(x=k_line, height=pi_truth_k,
            color='white', edgecolor='red', linestyle='--', label='true model', zorder=0) # 真の分布
    plt.bar(x=k_line, height=trace_pi[n], 
            alpha=0.6, color='purple', label='predict', zorder=1) # 予測分布
    if n > 0: # 初回は除く
        k_num, = np.where(s_nk[n-1] == 1) # n番目の次元番号を抽出
        plt.scatter(x=k_num.item() + 1, y=0.0, s=100, label='data', zorder=2) # 観測データ
    plt.xlabel('k')
    plt.ylabel('prob')
    plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
    plt.suptitle('Categorical Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{\pi}_{*}=(' + ', '.join([str(pi) for pi in np.round(trace_pi[n], 3)]) + ')$', loc='left')
    plt.ylim(-0.01, 1.0) # y軸の表示範囲
    plt.legend() # 凡例

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_2_2_Predict.gif")

#%%

print('end')

