
# ch 3.4.2 多次元ガウス分布：精度が未知の場合 ----------------------------------------------

# 事後分布 --------------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvtnorm)
library(mvnfast)


# データ数を指定
N <- 50
D <- 2 # (固定)

# 観測モデルのパラメータを指定
mu_d <- c(25, 50)
sigma_truth_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_truth_dd <- solve(sigma_truth_dd^2)

# 事前分布のパラメータを指定
W_dd <- matrix(c(0.00005, 0, 0, 0.00005), nrow = 2, ncol = 2)
nu <- 2


# 作図用の点を生成
x_vec <- seq(mu_d[1] - 2 * sigma_truth_dd[1, 1], mu_d[1] + 2 * sigma_truth_dd[1, 1], by = 0.5)
y_vec <- seq(mu_d[2] - 2 * sigma_truth_dd[2, 2], mu_d[2] + 2 * sigma_truth_dd[2, 2], by = 0.5)
point_df <- tibble(
  x = rep(x_vec, times = length(y_vec)), 
  y = rep(y_vec, each = length(x_vec))
)
mu_df <- tibble(
  x = mu_d[1], 
  y = mu_d[2]
)


# 2次元ガウス分布に従うデータを生成
x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_d, sigma = sigma_truth_dd^2)
summary(x_nd)

# 観測データのデータフレーム
sample_df <- tibble(
  x = x_nd[, 1], 
  y = x_nd[, 2]
)

# 観測モデルのデータフレーム
model_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = mu_d, sigma = sigma_truth_dd^2), # 確率密度
) %>% 
  dplyr::select(density) %>% 
  cbind(point_df, .)

# 観測データの散布図を作成
ggplot() + 
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..)) + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均パラメータ
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=(", paste(round(mu_d, 1), collapse = ", "), ")", 
                         ", sigma=(", paste(round(sigma_truth_dd, 1), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# 事後分布のパラメータを計算
W_hat_dd <- solve(
  (t(x_nd) - mu_d) %*% t(t(x_nd) - mu_d) + solve(W_dd)
)
nu_hat <- N + nu

# 精度パラメータの期待値を計算
lambda_E_dd <- nu_hat * W_hat_dd

# 事後分布の期待値を用いた分布を計算
posterior_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = mu_d, sigma = solve(lambda_E_dd)) # 確率密度
) %>% 
  dplyr::select(density) %>% 
  cbind(point_df, .)

# 作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x, y, z = density, color = ..level..)) + # 精度の期待値を用いた分布
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均値
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=(", paste(round(mu_d, 1), collapse = ", "), ")", 
                         ", E_sigma_hat=(", paste(round(sqrt(solve(lambda_E_dd)), 1), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# 予測分布 --------------------------------------------------------------------

# 予測分布のパラメータを計算
mu_s_d <- mu_d
lambda_s_hat_dd <- (1 - D + nu_hat) * W_hat_dd
nu_s_hat <- 1 - D + nu_hat

# 予測分布を計算
predict_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvt(x = xy, delta = mu_s_d, sigma = solve(lambda_s_hat_dd), df = nu_s_hat) # 確率密度
) %>% 
  dplyr::select(density) %>% 
  cbind(point_df, .)
predict_df <- cbind(
  point_df, 
  density = mvnfast::dmvt(X = as.matrix(point_df), mu = mu_s_d, sigma = solve(lambda_s_hat_dd), df = nu_s_hat) # 確率密度
)

# 作図
ggplot() + 
  geom_contour(data = predict_df, aes(x, y, z = density, color = ..level..)) + # 予測分布
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均パラメータ
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu_s=(", paste(round(mu_s_d, 1), collapse = ", "), ")", 
                         ", lambda_s_hat=(", paste(round(lambda_s_hat_dd, 1), collapse = ", "), ")", 
                         ", df=", nu_s_hat), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# gif ---------------------------------------------------------------------

# 追加パッケージ
library(gganimate)


# データ数を指定
N <- 100
D <- 2 # (固定)

# 観測モデルのパラメータを指定
mu_d <- c(25, 50)
sigma_truth_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_truth_dd <- solve(sigma_truth_dd^2)

# 事前分布のパラメータを指定
nu <- 2
W_dd <- matrix(c(0.00005, 0, 0, 0.00005), nrow = 2, ncol = 2)

# 精度パラメータの期待値を計算
lambda_E_dd <- nu * W_dd


# 作図用の点を生成
x_vec <- seq(mu_d[1] - 3 * sigma_truth_dd[1, 1], mu_d[1] + 3 * sigma_truth_dd[1, 1], by = 1)
y_vec <- seq(mu_d[2] - 3 * sigma_truth_dd[2, 2], mu_d[2] + 3 * sigma_truth_dd[2, 2], by = 1)
point_df <- tibble(
  x = rep(x_vec, times = length(y_vec)), 
  y = rep(y_vec, each = length(x_vec))
)
mu_df <- tibble(
  x = mu_d[1], 
  y = mu_d[2]
)

# 観測モデルを計算
model_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = mu_d, sigma = sigma_truth_dd^2), # 確率密度
) %>% 
  dplyr::select(density) %>% 
  cbind(point_df, .)


# 事前分布の期待値を用いた分布を計算
posterior_df <- tidyr::tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = mu_d, sigma = solve(lambda_E_dd)), # 確率密度
  iteration = 0 # 試行回数
) %>% 
  dplyr::select(density, iteration) %>% 
  cbind(point_df, .)


# 予測分布のパラメータを計算
mu_s_d <- mu_d
lambda_s_dd <- (1 - D + nu) * W_dd
nu_s <- 1 - D + nu

# 初期値による予測分布を計算
predict_df <- cbind(
  point_df, 
  density = mvnfast::dmvt(X = as.matrix(point_df), mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s), # 確率密度
  iteration = 0 # 試行回数
)

# ベイズ推論
for(i in 1:N) {
  
  # 2次元ガウス分布に従うデータを生成
  x_nd <- mvtnorm::rmvnorm(n = 1, mean = mu_d, sigma = sigma_truth_dd^2)
  
  # 観測データを記録
  if(i > 1) { # 初回以外
    # オブジェクトを結合
    x_mat <- rbind(x_mat, x_nd)
    sample_df <- tibble(
      x = x_mat[, 1],
      y = x_mat[, 2], 
      iteration = i
    ) %>% 
      rbind(sample_df, .)
  } else if(i == 1){ # 初回
    # オブジェクトを作成
    x_mat <- x_nd
    sample_df <- tibble(
      x = x_mat[, 1],
      y = x_mat[, 2], 
      iteration = i
    )
  }
  
  # 事後分布のパラメータを更新
  W_dd <- solve(
    (t(x_nd) - mu_d) %*% t(t(x_nd) - mu_d) + solve(W_dd)
  )
  nu <- 1 + nu
  
  # 精度パラメータの期待値を計算
  lambda_E_dd <- nu * W_dd
  
  # 事後分布の期待値を用いた分布を計算
  tmp_posterior_df <- tidyr::tibble(
    xy = point_df, 
    density = mvtnorm::dmvnorm(x = xy, mean = mu_d, sigma = solve(lambda_E_dd)), # 確率密度
    iteration = i # 試行回数
  ) %>% 
    dplyr::select(density, iteration) %>% 
    cbind(point_df, .)
  
  # 予測分布のパラメータを更新
  mu_s_d <- mu_d
  lambda_s_dd <- (1 - D + nu) * W_dd
  nu_s <- 1 - D + nu
  
  # 予測分布を計算
  tmp_predict_df <- cbind(
    point_df, 
    density = mvnfast::dmvt(X = as.matrix(point_df), mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s), # 確率密度
    iteration = i # 試行回数
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
  
  # 動作確認
  print(i)
}


# 事後分布の期待値を用いた分布を作図
posterior_graph <- ggplot() + 
  geom_contour(data = posterior_df, aes(x, y, z = density, color = ..level..)) +  # 精度の期待値を用いた分布
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) +  # 平均パラメータ
  transition_manual(iteration) +  # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "N={current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル

# gif画像を作成
animate(posterior_graph, nframes = N + 1, fps = 5)


# 予測分布を作図
predict_graph <- ggplot() + 
  geom_contour(data = predict_df, aes(x, y, z = density, color = ..level..)) +  # 予測分布
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) +  # 平均パラメータ
  transition_manual(iteration) +  # フレーム
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = "N={current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル

# gif画像を作成
animate(predict_graph, nframes = N + 1, fps = 5)


