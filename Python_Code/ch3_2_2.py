# 3.2.2 カテゴリ分布の学習と予測

#%%

# 3.2.2項で利用するライブラリ
import numpy as np
import math # 対数ガンマ関数:lgamma()
#from scipy.stats import dirichlet # ディリクレ分布
import matplotlib.pyplot as plt

#%%

## 真のモデルの設定

# 次元数:(固定)
K = 3

# 真のパラメータを指定
pi_truth_k = np.array([0.3, 0.5, 0.2])

# x軸の値を作成
k_line = np.arange(1, K + 1)

#%%

# 全てのパターンのデータを作成
s_kk = np.identity(K)

# 確率を計算:式(2.29)
true_model = np.prod(pi_truth_k**s_kk, axis=1)

# 確率を計算:SciPy ver
from scipy.stats import multinomial # 多項分布
true_model = multinomial.pmf(x=s_kk, n=1, p=pi_truth_k)
print(true_model)


# 尤度を作図
plt.bar(x=k_line, height=pi_truth_k, color='purple') # 真のモデル
plt.xlabel('k')
plt.ylabel('prob')
plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$\pi=(' + ', '.join([str(k) for k in  pi_truth_k]) + ')$', loc='left')
plt.ylim(0, 1)
plt.show()

#%%

## 観測データの生成

# データ数を指定
N = 50

# (観測)データを生成
s_nk = np.random.multinomial(n=1, pvals=pi_truth_k, size=N)

# 観測のデータを確認
print(np.sum(s_nk, axis=0))

#%%

# 観測データのヒストグラムを作図
plt.bar(x=k_line, height=np.sum(s_nk, axis=0)) # 観測データ
plt.xlabel('k')
plt.ylabel('count')
plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \pi=(' + ', '.join([str(k) for k in pi_truth_k]) + ')$', loc='left')
plt.show()

#%%

## 事前分布の設定

# 事前分布のパラメータを指定
alpha_k = np.array([1.0, 1.0, 1.0])

#%%

# 作図用の点を設定
point_vec = np.arange(0.0, 1.001, 0.02)

# 格子状の点を作成
X, Y, Z = np.meshgrid(point_vec, point_vec, point_vec)

# 確率密度の計算用にまとめる
pi_point = np.array([list(X.flatten()), list(Y.flatten()), list(Z.flatten())]).T
pi_point = pi_point[1:, :] # (0, 0, 0)の行を除去
pi_point /= np.sum(pi_point, axis=1, keepdims=True) # 正規化
pi_point = np.unique(pi_point, axis=0) # 重複を除去

#%%

# 事前分布(ディリクレ分布)の確率密度を計算:式(2.41)
ln_C_dir = math.lgamma(np.sum(alpha_k)) - np.sum([math.lgamma(a) for a in alpha_k]) # 正規化項(対数)
prior = np.exp(ln_C_dir) * np.prod(pi_point**(alpha_k - 1), axis=1)

# 事前分布(ディリクレ分布)の確率密度を計算:SciPy ver
#prior = np.array([
#    dirichlet.pdf(x=pi_point[i], alpha=alpha_k) for i in range(len(pi_point))
#])

#%%

# 三角座標に変換
tri_x = pi_point[:, 1] + pi_point[:, 2] / 2
tri_y = np.sqrt(3) * pi_point[:, 2] / 2

# 事前分布を作図
plt.scatter(tri_x, tri_y, c=prior, cmap='jet') # 事前分布
plt.xlabel('$\pi_1, \pi_2$') # x軸ラベル
plt.ylabel('$\pi_1, \pi_3$') # y軸ラベル
plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)']) # x軸目盛
plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)']) # y軸目盛
plt.suptitle('Dirichlet Distribution', fontsize=20)
plt.title('$\\alpha=(' + ', '.join([str(k) for k in alpha_k]) + ')$', loc='left')
plt.colorbar() # 凡例
plt.gca().set_aspect('equal') # アスペクト比
plt.show()

#%%

## 事後分布の計算

# 事後分布のパラメータを計算:式(3.28)
alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
print(alpha_hat_k)

# 事後分布(ディリクレ分布)の確率密度を計算:式(2.41)
ln_C_dir = math.lgamma(np.sum(alpha_hat_k)) - np.sum([math.lgamma(a) for a in alpha_hat_k]) # 正規化項(対数)
posterior = np.prod(pi_point**(alpha_hat_k - 1), axis=1)
posterior *= np.exp(ln_C_dir)

# 事後分布(ディリクレ分布)の確率密度を計算:SciPy ver
#posterior = np.array([
#    dirichlet.pdf(x=pi_point[i], alpha=alpha_hat_k) for i in range(len(pi_point))
#])

#%%

# 真のパラメータの値を三角座標に変換
tri_x_truth = pi_truth_k[1] + pi_truth_k[2] / 2
tri_y_truth = np.sqrt(3) * pi_truth_k[2] / 2

# 事後分布を作図
plt.scatter(tri_x, tri_y, c=posterior, cmap='jet') # 事後分布
plt.xlabel('$\pi_1, \pi_2$') # x軸ラベル
plt.ylabel('$\pi_1, \pi_3$') # y軸ラベル
plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)']) # x軸目盛
plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)']) # y軸目盛
plt.suptitle('Dirichlet Distribution', fontsize=20)
plt.title('$\\alpha=(' + ', '.join([str(k) for k in alpha_hat_k]) + ')$', loc='left')
plt.colorbar() # 凡例
plt.gca().set_aspect('equal') # アスペクト比
plt.scatter(tri_x_truth, tri_y_truth, marker='x', color='black', s=200) # 真のパラメータ
plt.show()

#%%

## 予測分布の計算

# 予測分布のパラメータを計算
pi_hat_star_k = alpha_hat_k / np.sum(alpha_hat_k)
pi_hat_star_k = (np.sum(s_nk, axis=0) + alpha_k) / np.sum(np.sum(s_nk, axis=0) + alpha_k)
print(pi_hat_star_k)

#%%

# 予測分布を作図
plt.bar(x=k_line, height=pi_truth_k, label='truth',
        alpha=0.5, color='white', edgecolor='red', linestyle='dashed') # 真のモデル
plt.bar(x=k_line, height=pi_hat_star_k, label='predict', 
        alpha=0.5, color='purple') # 予測分布
plt.xlabel('k')
plt.ylabel('prob')
plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\pi}_{*}=(' + ', '.join([str(k) for k in np.round(pi_hat_star_k, 2)]) + ')$', loc='left')
plt.ylim(0, 1)
plt.show()

#%%

### アニメーション

# 利用するライブラリ
import numpy as np
from scipy.stats import dirichlet # ディリクレ分布
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

# 描画用の点を設定
point_vec = np.arange(0.0, 1.001, 0.025)

# 格子状の点を作成
X, Y, Z = np.meshgrid(point_vec, point_vec, point_vec)

# 確率密度の計算用にまとめる
pi_point = np.array([list(X.flatten()), list(Y.flatten()), list(Z.flatten())]).T
pi_point = pi_point[1:, :] # (0, 0, 0)の行を除去
pi_point /= np.sum(pi_point, axis=1, keepdims=True) # 正規化
pi_point = np.unique(pi_point, axis=0) # 重複を除去

# 三角座標に変換
tri_x = pi_point[:, 1] + pi_point[:, 2] / 2
tri_y = np.sqrt(3) * pi_point[:, 2] / 2

#%%

## 推論処理

# 次元数:(固定)
K = 3

# 真のパラメータを指定
pi_truth_k = np.array([0.3, 0.5, 0.2])

# 事前分布のパラメータを指定
alpha_k = np.array([1.0, 1.0, 1.0])

# 初期値による予測分布のパラメータを計算
mu_star_k = alpha_k / np.sum(alpha_k)

# データ数を指定
N = 100

# 記録用の受け皿を初期化
s_nk = np.empty((N, K))
trace_alpha = [list(alpha_k)]
trace_posterior = [[dirichlet.pdf(x=pi_point[i], alpha=alpha_k) for i in range(len(pi_point))]]
trace_predict = [list(mu_star_k)]

# ベイズ推論
for n in range(N):
    # (観測)データを生成
    s_nk[n] = np.random.multinomial(n=1, pvals=pi_truth_k, size=1)[0]
    
    # 事後分布のパラメータを更新
    alpha_k += s_nk[n]
    
    # 値を記録
    trace_alpha.append(list(alpha_k))
    
    # 事後分布(ディリクレ分布)の確率密度を計算
    trace_posterior.append(
        [dirichlet.pdf(x=pi_point[i], alpha=alpha_k) for i in range(len(pi_point))]
    )
    
    # 予測分布のパラメータを更新
    mu_star_k = alpha_k / np.sum(alpha_k)
    
    # 予測分布(カテゴリ分布)の確率を記録
    trace_predict.append(list(mu_star_k))
    
    # 途中経過を表示
    print('n=' + str(n + 1) + ' (' + str(np.round((n + 1) / N * 100, 1)) + '%)')

# 観測のデータを確認
print(np.sum(s_nk, axis=0))

#%%

## 事後分布の推移をgif画像化

# 真のパラメータの値を三角座標に変換
tri_x_truth = pi_truth_k[1] + pi_truth_k[2] / 2
tri_y_truth = np.sqrt(3) * pi_truth_k[2] / 2

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の事後分布を作図
    plt.scatter(tri_x, tri_y, c=trace_posterior[n], cmap='jet') # 事後分布
    plt.scatter(tri_x_truth, tri_y_truth, marker='x', color='black', s=200) # 真のパラメータ
    plt.xlabel('$\pi_1, \pi_2$') # x軸ラベル
    plt.ylabel('$\pi_1, \pi_3$') # y軸ラベル
    plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)']) # x軸目盛
    plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)']) # y軸目盛
    plt.suptitle('Dirichlet Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\\alpha}=(' + ', '.join([str(a) for a in trace_alpha[n]]) + ')$', loc='left')
    plt.gca().set_aspect('equal') # アスペクト比

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=N + 1, interval=100)
posterior_anime.save("ch3_2_2_Posterior.gif")

#%%

## 予測分布の推移をgif画像化

# x軸の値を作成
k_line = np.arange(1, K + 1)

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.bar(x=k_line, height=pi_truth_k, label='truth',
            alpha=0.5, color='white', edgecolor='red', linestyle='dashed') # 真の分布
    plt.bar(x=k_line, height=trace_predict[n], label='predict', 
            alpha=0.5, color='purple') # 予測分布
    plt.xlabel('k')
    plt.ylabel('prob')
    plt.xticks(ticks=k_line, labels=k_line) # x軸目盛
    plt.suptitle('Categorical Distribution', fontsize=20)
    plt.title('$N=' + str(n) + ', \hat{\pi}_{*}=(' + ', '.join([str(k) for k in np.round(trace_predict[n], 2)]) + ')$', loc='left')
    plt.ylim(0, 1)
    plt.legend()

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_2_2_Predict.gif")

#%%

print('end')

