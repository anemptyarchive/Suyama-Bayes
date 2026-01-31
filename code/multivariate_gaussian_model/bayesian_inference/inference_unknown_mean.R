
# 多次元ガウスモデル ------------------------------------------------------------

# chapter 3.4.1
# 平均が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### ・生成分布(多次元ガウス分布)の設定 -----

# 次元数を指定
D <- 2

# 真の平均ベクトルを指定
mu_truth_d <- c(25, 50)

# (既知の)分散共分散行列を指定
sigma_dd <- matrix(c(900, -100, -100, 400), nrow = D, ncol = D)

# (既知の)精度行列を計算
lambda_dd <- solve(sigma_dd)


# グラフ用のxの値を作成
x_1_vec <- seq(
  mu_truth_d[1] - sqrt(sigma_dd[1, 1]) * 3, 
  mu_truth_d[1] + sqrt(sigma_dd[1, 1]) * 3, 
  length.out = 301
)
x_2_vec <- seq(
  mu_truth_d[2] - sqrt(sigma_dd[2, 2]) * 3, 
  mu_truth_d[2] + sqrt(sigma_dd[2, 2]) * 3, 
  length.out = 301
)

# グラフ用のxの点を作成
x_mat <- tidyr::expand_grid(
  x_1 = x_1_vec, 
  x_2 = x_2_vec
) |> # 格子点を作成
  as.matrix() # マトリクスに変換


# 真の分布(多次元ガウス分布)を計算:式(2.72)
model_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = x_mat, mu = mu_truth_d, sigma = sigma_dd) # 確率密度
)

# パラメータラベル用の文字列を作成
model_param_text <- paste0(
  "list(mu==(list(", paste(round(mu_truth_d, 1), collapse = ", "), "))", 
  ", Lambda==(list(", paste(round(lambda_dd, 5), collapse = ", "), ")))"
)

# 真の分布を作図
ggplot() + 
  #geom_contour(data = model_df, mapping = aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 真の分布:(等高線図)
  geom_contour_filled(data = model_df, mapping = aes(x = x_1, y = x_2, z = dens, fill = ..level..), alpha = 0.8) + # 真の分布:(塗りつぶし等高線図)
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = model_param_text), 
       color = "density", fill = "density", 
       x = expression(x[1]), y = expression(x[2]))


### ・データの生成 -----

# (観測)データ数を指定
N <- 100


# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_truth_d, sigma = sigma_dd)

# 観測データを格納
data_df <- tibble::tibble(
  x_1 = x_nd[, 1], # x軸の値
  x_2 = x_nd[, 2]  # y軸の値
)

# パラメータラベル用の文字列を作成
sample_param_text <- paste0(
  "list(mu==(list(", paste(round(mu_truth_d, 1), collapse = ", "), "))", 
  ", Lambda==(list(", paste(round(lambda_dd, 5), collapse = ", "), "))", 
  ", N==", N, ")"
)

# 観測データの散布図を作成
ggplot() + 
  #geom_contour(data = model_df, aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 真の分布:(等高線図)
  geom_contour_filled(data = model_df, aes(x = x_1, y = x_2, z = dens, fill = ..level..), alpha = 0.8) + # 真の分布:(塗りつぶし等高線図)
  geom_point(data = data_df, aes(x = x_1, y = x_2), color = "orange") + # 観測データ
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = sample_param_text), 
       color = "density", fill = "density", 
       x = expression(x[1]), y = expression(x[2]))


### ・事前分布(多次元ガウス分布)の設定 -----

# μの事前分布の平均ベクトルを指定
m_d <- rep(0, times = D)

# μの事前分布の分散共分散行列を指定
sigma_mu_dd <- diag(D) * 10000

# μの事前分布の精度行列を計算
lambda_mu_dd <- solve(sigma_mu_dd)


# グラフ用のμの値を作成
mu_1_vec <- seq(
  mu_truth_d[1] - sqrt(sigma_mu_dd[1, 1]), 
  mu_truth_d[1] + sqrt(sigma_mu_dd[1, 1]), 
  length.out = 301
)
mu_2_vec <- seq(
  mu_truth_d[2] - sqrt(sigma_mu_dd[2, 2]), 
  mu_truth_d[2] + sqrt(sigma_mu_dd[2, 2]), 
  length.out = 301
)

# グラフ用のμの点を作成
mu_mat <- tidyr::expand_grid(
  mu_1 = mu_1_vec, 
  mu_2 = mu_2_vec  
) |> # 格子点を作成
  as.matrix() # マトリクスに変換

# 真のμを格納
param_df <- tibble::tibble(
  mu_1 = mu_truth_d[1], # x軸の値
  mu_2 = mu_truth_d[2]  # y軸の値
)


# μの事前分布(多次元ガウス分布)を計算:式(2.72)
prior_df <- tibble::tibble(
  mu_1 = mu_mat[, 1], # x軸の値
  mu_2 = mu_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = mu_mat, mu = m_d, sigma = sigma_mu_dd) # 確率密度
)

# パラメータラベル用の文字列を作成
prior_param_text <- paste0(
  "list(m==(list(", paste(round(m_d, 1), collapse = ", "), "))", 
  ", Lambda[mu]==(list(", paste(round(lambda_mu_dd, 5), collapse = ", "), ")))"
)

# μの事前分布を作図
ggplot() + 
  #geom_contour(data = prior_df, mapping = aes(x = mu_1, y = mu_2, z = dens, color = ..level..)) + # μの事前分布:(等高線図)
  geom_contour_filled(data = prior_df, mapping = aes(x = mu_1, y = mu_2, z = dens, fill = ..level..), alpha = 0.8) + # μの事前分布:(塗りつぶし等高線図)
  geom_point(data = param_df, mapping = aes(x = mu_1, y = mu_2, shape = "param"), 
             color = "red", size = 6) + # 真のμ
  scale_shape_manual(breaks = "param", values = 4, labels = "true parameter", name = "") + # (凡例表示用の黒魔術)
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = prior_param_text), 
       color = "density", fill = "density", 
       x = expression(mu[1]), y = expression(mu[2]))


### ・事後分布(多次元ガウス分布)の計算 -----

# μの事後分布の精度行列を計算:式(3.102)
lambda_mu_hat_dd <- N * lambda_dd + lambda_mu_dd

# μの事後分布の平均ベクトルを計算:式(3.103)
m_hat_d <- (solve(lambda_mu_hat_dd) %*% (lambda_dd %*% colSums(x_nd) + lambda_mu_dd %*% m_d)) |> 
  as.vector()


# μの事後分布(多次元ガウス分布)を計算:式(2.72)
posterior_df <- tibble::tibble(
  mu_1 = mu_mat[, 1], # x軸の値
  mu_2 = mu_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = mu_mat, mu = m_hat_d, sigma = solve(lambda_mu_hat_dd)) # 確率密度
)

# パラメータラベル用の文字列を作成
posterior_param_text <- paste0(
  "list(N ==", N, 
  ", hat(m)==(list(", paste(round(m_hat_d, 1), collapse = ", "), "))", 
  ", hat(Lambda)[mu]==(list(", paste(round(lambda_mu_hat_dd, 5), collapse = ", "), ")))"
)

# μの事後分布を作図
ggplot() + 
  #geom_contour(data = posterior_df, mapping = aes(x = mu_1, y = mu_2, z = dens, color = ..level..)) + # μの事後分布:(等高線図)
  geom_contour_filled(data = posterior_df, mapping = aes(x = mu_1, y = mu_2, z = dens, fill = ..level..), alpha = 0.8) + # μの事後分布:(塗りつぶし等高線図)
  geom_point(data = param_df, mapping = aes(x = mu_1, y = mu_2, shape = "param"), 
             color = "red", size = 6) + # 真のμ
  scale_shape_manual(breaks = "param", values = 4, labels = "true parameter", name = "") + # (凡例表示用の黒魔術)
  coord_cartesian(xlim = c(min(mu_1_vec), max(mu_1_vec)), ylim = c(min(mu_2_vec), max(mu_2_vec))) + # 表示範囲
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = posterior_param_text), 
       color = "density", fill = "density", 
       x = expression(mu[1]), y = expression(mu[2]))


### ・予測分布(多次元ガウス分布)の計算 -----

# 予測分布の平均ベクトルを計算:式(3.110')
mu_s_hat_d <- m_hat_d

# 予測分布の精度行列を計算:式(3.109')
lambda_s_hat_dd <- solve(solve(lambda_dd) + solve(lambda_mu_hat_dd))


# 予測分布(多次元ガウス分布)を計算:式(2.72)
predict_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = x_mat, mu = mu_s_hat_d, sigma = solve(lambda_s_hat_dd)) # 確率密度
)

# パラメータラベル用の文字列を作成
predict_param_text <- paste0(
  "list(N==", N, 
  ", hat(mu)[s]==(list(", paste(round(mu_s_hat_d, 1), collapse = ", "), "))", 
  ", hat(Lambda)[s]==(list(", paste(round(lambda_s_hat_dd, 5), collapse = ", "), ")))"
)

# 予測分布を作図
ggplot() + 
  geom_contour(data = model_df, mapping = aes(x = x_1, y = x_2, z = dens, color = ..level..), 
               alpha = 1, linetype = "dashed") + # 真の分布
  #geom_contour(data = predict_df, mapping = aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 予測分布:(等高線図)
  geom_contour_filled(data = predict_df, mapping = aes(x = x_1, y = x_2, z = dens, fill = ..level..), alpha = 0.8) + # 予測分布:(塗りつぶし等高線図)
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = predict_param_text), 
       color = "density", fill = "density", 
       x = expression(x[1]), y = expression(x[2]))


