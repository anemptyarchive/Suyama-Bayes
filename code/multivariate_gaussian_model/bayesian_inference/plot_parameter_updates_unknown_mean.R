
# 多次元ガウスモデル ------------------------------------------------------------

# chapter 3.4.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# アニメーションによる学習推移の可視化 --------------------------------------------------------------

### ・モデルの設定 -----

# 次元数を指定
D <- 2

# 真の平均ベクトルを指定
mu_truth_d <- c(25, 50)

# (既知の)分散共分散行列を指定
sigma_dd <- matrix(c(900, -100, -100, 400), nrow = D, ncol = D)

# (既知の)精度行列を計算
lambda_dd <- solve(sigma_dd)

# μの事前分布の平均ベクトルを指定
m_d <- rep(0, times = D)

# μの事前分布の分散共分散行列を指定
sigma_mu_dd <- diag(D) * 10000

# μの事前分布の精度行列を計算
lambda_mu_dd <- solve(sigma_mu_dd)

# データ数(試行回数)を指定
N <- 100


# グラフ用のμの値を作成
mu_1_vec <- seq(
  mu_truth_d[1] - sqrt(sigma_mu_dd[1, 1]), 
  mu_truth_d[1] + sqrt(sigma_mu_dd[1, 1]), 
  length.out = 201
)
mu_2_vec <- seq(
  mu_truth_d[2] - sqrt(sigma_mu_dd[2, 2]), 
  mu_truth_d[2] + sqrt(sigma_mu_dd[2, 2]), 
  length.out = 201
)

# グラフ用のμの点を作成
mu_mat <- tidyr::expand_grid(mu_1 = mu_1_vec, mu_2 = mu_2_vec ) |> # 格子点を作成
  as.matrix() # マトリクスに変換

# グラフ用のxの値を作成
x_1_vec <- seq(
  mu_truth_d[1] - sqrt(sigma_dd[1, 1]) * 3, 
  mu_truth_d[1] + sqrt(sigma_dd[1, 1]) * 3, 
  length.out = 201
)
x_2_vec <- seq(
  mu_truth_d[2] - sqrt(sigma_dd[2, 2]) * 3, 
  mu_truth_d[2] + sqrt(sigma_dd[2, 2]) * 3, 
  length.out = 201
)

# グラフ用のxの点を作成
x_mat <- tidyr::expand_grid(x_1 = x_1_vec, x_2 = x_2_vec) |> # 格子点を作成
  as.matrix() # マトリクスに変換


### ・推論処理：for関数による処理 -----

# μの事前分布(多次元ガウス分布)を計算:式(2.72)
anime_posterior_df <- tibble::tibble(
  mu_1 = mu_mat[, 1], 
  mu_2 = mu_mat[, 2], 
  dens = mvnfast::dmvn(X = mu_mat, mu = m_d, sigma = solve(lambda_mu_dd)), # 確率密度
  param = paste0(
    "N=", 0, 
    ", m=(", paste(round(m_d, 1), collapse = ", "), ")", 
    ", lambda_mu=(", paste(round(lambda_mu_dd, 5), collapse = ", "), ")"
  ) |> 
  as.factor() # フレーム切替用ラベル
)


# 初期値による予測分布のパラメータを計算:式(3.109-10)
mu_s_d      <- m_d
lambda_s_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))

# 予測分布(多次元ガウス分布)を計算:式(2.72)
anime_predict_df <- tibble::tibble(
  x_1 = x_mat[, 1], 
  x_2 = x_mat[, 2], 
  dens = mvnfast::dmvn(X = x_mat, mu = mu_s_d, sigma = solve(lambda_s_dd)), # 確率密度
  param = paste0(
    "N=", 0, 
    ", mu_s=(", paste(round(mu_s_d, 1), collapse = ", "), ")", 
    ", lambda_s=(", paste(round(lambda_s_dd, 5), collapse = ", "), ")"
  ) |> 
  as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
x_nd <- matrix(NA, nrow = N, ncol = D)

# ベイズ推論
for(n in 1:N) {
  
  # 多次元ガウス分布に従うデータを生成
  x_nd[n, ] <- mvnfast::rmvn(n = 1, mu = mu_truth_d, sigma = sigma_dd) |> 
    as.vector()
  
  # μの事後分布のパラメータを更新:式(3.102-3)
  old_lambda_mu_dd <- lambda_mu_dd
  lambda_mu_dd     <- lambda_dd + lambda_mu_dd
  m_d <- solve(lambda_mu_dd) %*% (lambda_dd %*% x_nd[n, ] + old_lambda_mu_dd %*% m_d) |> 
    as.vector()
  
  # μの事後分布(多次元ガウス分布)を計算:式(2.72)
  tmp_posterior_df <- tibble::tibble(
    mu_1 = mu_mat[, 1], 
    mu_2 = mu_mat[, 2], 
    dens = mvnfast::dmvn(X = mu_mat, mu = m_d, sigma = solve(lambda_mu_dd)), # 確率密度
    param = paste0(
      "N=", n, 
      ", m=(", paste(round(m_d, 1), collapse = ", "), ")", 
      ", lambda_mu=(", paste(round(lambda_mu_dd, 5), collapse = ", "), ")"
    ) |> 
    as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメータを更新:式(3.109-10)
  mu_s_d      <- m_d
  lambda_s_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))
  
  # 予測分布(多次元ガウス分布)を計算:式(2.72)
  tmp_predict_df <- tibble::tibble(
    x_1 = x_mat[, 1], 
    x_2 = x_mat[, 2], 
    dens = mvnfast::dmvn(X = x_mat, mu = mu_s_d, sigma = solve(lambda_s_dd)), # 確率密度
    param = paste0(
      "N=", n, 
      ", mu_s=(", paste(round(mu_s_d, 1), collapse = ", "), ")", 
      ", lambda_s=(", paste(round(lambda_s_dd, 5), collapse = ", "), ")"
    ) |> 
    as.factor() # フレーム切替用ラベル
  )
  
  # 推論結果を結合
  anime_posterior_df <- rbind(anime_posterior_df, tmp_posterior_df)
  anime_predict_df   <- rbind(anime_predict_df, tmp_predict_df)
  
  # 動作確認
  print(paste0("n=", n, " (", round(n / N * 100, 1), "%)"))
}

# 観測データを確認
summary(x_nd)


### ・推論処理：tidyverseパッケージによる処理 -----

# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_truth_d, sigma = sigma_dd)

# 試行ごとにμの事後分布を計算
anime_posterior_df <- tidyr::expand_grid(
  n = 0:N, # データ番号(試行回数)
  mu_1 = mu_1_vec, 
  mu_2 = mu_2_vec
) |> # 試行ごとに格子点を複製
  dplyr::group_by(n) |> # 分布の計算用にグループ化
  dplyr::mutate(
    # μの事後分布のパラメータを計算:式(3.102-3)
    lambda_mu_lt = (unique(n) * lambda_dd + lambda_mu_dd) |> 
      list(), # リストに格納
    m_lt = dplyr::if_else(
      n > 0, # 事前分布を除く
      true = (solve(lambda_mu_lt[[1]]) %*% (lambda_dd %*% colSums(x_nd[0:unique(n), , drop = FALSE]) + lambda_mu_dd %*% m_d)) |> 
        as.vector() |> 
        list(), # リストに格納
      false = m_d |> 
        list() # リストに格納
    ), 
    
    # μの事後分布(多次元ガウス分布)を計算:式(2.72)
    dens = mvnfast::dmvn(
      X = mu_mat, 
      mu = m_lt[[1]], 
      sigma = solve(lambda_mu_lt[[1]])
    ), # 確率密度
    param = paste0(
      "N=", unique(n), 
      ", m=(", paste(round(m_lt[[1]], 1), collapse = ", "), ")", 
      ", lambda_mu=(", paste(round(lambda_mu_lt[[1]], 5), collapse = ", "), ")"
    ) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup() # グループ化を解除

# 試行ごとに予測分布を計算
anime_predict_df <- tidyr::expand_grid(
  n = 0:N, # データ番号(試行回数)
  x_1 = x_1_vec, 
  x_2 = x_2_vec
) |> # 試行ごとに格子点を複製
  dplyr::group_by(n) |> # 分布の計算用にグループ化
  dplyr::mutate(
    # μの事後分布のパラメータを計算:式(3.102)
    lambda_mu_lt = (unique(n) * lambda_dd + lambda_mu_dd) |> 
      list(), # リストに格納
    
    # 予測分布のパラメータを計算:式(3.103,9')
    mu_s_lt = dplyr::if_else(
      n > 0, # 事前分布を除く
      true = (solve(lambda_mu_lt[[1]]) %*% (lambda_dd %*% colSums(x_nd[0:unique(n), , drop = FALSE]) + lambda_mu_dd %*% m_d)) |> 
        as.vector() |> 
        list(), # リストに格納
      false = m_d |> 
        list() # リストに格納
    ), 
    sigma_s_lt = (solve(lambda_dd) + solve(lambda_mu_lt[[1]])) |> 
      list(), 
    
    # 予測分布(多次元ガウス分布)を計算:式(2.72)
    dens = mvnfast::dmvn(
      X = x_mat, 
      mu = mu_s_lt[[1]], 
      sigma = sigma_s_lt[[1]]
    ), # 確率密度
    param = paste0(
      "N=", unique(n), 
      ", mu_s=(", paste(round(mu_s_lt[[1]], 1), collapse = ", "), ")", 
      ", lambda_s=(", paste(round(solve(sigma_s_lt[[1]]), 5), collapse = ", "), ")"
    ) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup() # グループ化を解除


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x_1 = c(NA, x_nd[, 1]), # x軸の値
  x_2 = c(NA, x_nd[, 2]), # y軸の値
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# 真のμを格納
param_df <- tibble::tibble(
  mu_1 = mu_truth_d[1], # x軸の値
  mu_2 = mu_truth_d[2]  # y軸の値
)

# μの事後分布のアニメーションを作図
anime_posterior_graph <- ggplot() + 
  geom_contour(data = anime_posterior_df, mapping = aes(x = mu_1, y = mu_2, z = dens, color = ..level..)) + # μの事後分布:(等高線図)
  #geom_contour_filled(data = anime_posterior_df, mapping = aes(x = mu_1, y = mu_2, z = dens, fill = ..level..), alpha = 0.8) + # μの事後分布:(塗りつぶし等高線図)
  geom_point(data = param_df, mapping = aes(x = mu_1, y = mu_2, shape = "param"), 
             color = "red", size = 6) + # 真のμ
  geom_point(data = anime_data_df, mapping = aes(x = x_1, y = x_2, shape = "data"), 
             color = "orange", size = 3) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_shape_manual(breaks = c("param", "data"), values = c(4, 19), 
                     labels = c("true parameter", "observation data"), name = "") + # (凡例表示用の黒魔術)
  guides(shape = guide_legend(override.aes = list(color = c("red", "orange"), size = c(6, 3), shape = c(4, 19)))) + # (凡例表示用の黒魔術)
  #coord_cartesian(xlim = c(min(mu_1_vec), max(mu_2_vec)), ylim = c(min(mu_2_vec), max(mu_2_vec))) + # 表示範囲
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "{current_frame}", 
       color = "density", fill = "density", 
       x = expression(mu[1]), y = expression(mu[2]))

# gif画像を作成
gganimate::animate(anime_posterior_graph, nframes = N+1+10, end_pause = 10, fps = 10, width = 800, height = 600)


# 観測データを格納
anime_data_df <- tibble::tibble(
  x_1 = c(NA, x_nd[, 1]), # x軸の値
  x_2 = c(NA, x_nd[, 2]), # y軸の値
  param = unique(anime_predict_df[["param"]]) # フレーム切替用ラベル
)

# 過去の観測データを格納
anime_alldata_df <- tidyr::expand_grid(
  frame = 1:N, # フレーム番号
  n = 1:N # 試行回数
) |> # 全ての組み合わせを作成
  dplyr::filter(n < frame) |> # フレーム番号以前のデータを抽出
  dplyr::mutate(
    x_1 = x_nd[n, 1], # x軸の値
    x_2 = x_nd[n, 2], # y軸の値
    param = unique(anime_predict_df[["param"]])[frame+1] # フレーム切替用ラベル
  )

# 真の分布(多次元ガウス分布)を計算:式(2.72)
model_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = x_mat, mu = mu_truth_d, sigma = sigma_dd) # 確率密度
)

# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = dens, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_contour(data = anime_predict_df, aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 予測分布:(等高線図)
  #geom_contour_filled(data = anime_predict_df, aes(x = x_1, y = x_2, z = dens, fill = ..level..), alpha = 0.8) + # 予測分布:(塗りつぶし等高線図)
  geom_point(data = anime_alldata_df, aes(x = x_1, y = x_2), 
             color = "orange", alpha = 0.5) + # 過去の観測データ
  geom_point(data = anime_data_df, aes(x = x_1, y = x_2), 
             color = "orange", size = 3) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "{current_frame}", 
       color = "density", fill = "density", 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(predict_graph, nframes = N+1+10, end_pause = 10, fps = 10, width = 800, height = 600)


