
# ch 3.4.1 多次元ガウス分布：平均が未知の場合 ----------------------------------------------

# 事後分布 --------------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvtnorm)


# データ数を指定
N <- 50

# 観測モデルのパラメータを指定
mu_truth_d <- c(25, 50)
sigma_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_dd <- solve(sigma_dd^2)

# 事前分布のパラメータを指定
m_d <- c(0, 0)
sigma_mu_dd <- matrix(c(100, 0, 0, 100), nrow = 2, ncol = 2)
lambda_mu_dd <- solve(sigma_mu_dd^2)


# 作図用の点を生成
x_vec <- seq(mu_truth_d[1] - 2 * sigma_dd[1, 1], mu_truth_d[1] + 2 * sigma_dd[1, 1], by = 0.25)
y_vec <- seq(mu_truth_d[2] - 2 * sigma_dd[2, 2], mu_truth_d[2] + 2 * sigma_dd[2, 2], by = 0.25)
point_df <- tibble(
  x = rep(x_vec, times = length(y_vec)), 
  y = rep(y_vec, each = length(x_vec))
)
mu_df <- tibble(
  x = mu_truth_d[1], 
  y = mu_truth_d[2]
)


# 2次元ガウス分布に従うデータを生成
x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_truth_d, sigma = solve(lambda_dd))
summary(x_nd)

# 観測データの散布図を作成
tibble(
  x = x_nd[, 1], 
  y = x_nd[, 2]
) %>% 
  ggplot(data = ., aes(x = x, y = y)) + 
    geom_point() + 
    geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + 
    labs(title = "Multivariate Gaussian Distribution", 
         subtitle = paste0("N=", N, ", mu=(", paste(round(mu_truth_d, 1), collapse = ", "), ")", 
                           ", sigma=(", paste(round(sigma_dd, 1), collapse = ", "), ")"), 
         x = expression(x[1]), y = expression(x[2]))


# 事後分布のパラメータを計算
lambda_mu_hat_dd <- N * lambda_dd + lambda_mu_dd
m_hat_d <- solve(lambda_mu_hat_dd) %*% (lambda_dd %*% colSums(x_nd) + lambda_mu_dd %*% m_d) %>% 
  as.vector()

# 事後分布を計算
posterior_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = m_hat_d, sigma = solve(lambda_mu_hat_dd)) # 確率密度
) %>% 
  dplyr::select(density) %>% 
  cbind(point_df, .)

# 作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x, y, z = density, color = ..level..)) + # 等高線
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 真の平均値
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=(", paste(round(m_hat_d, 1), collapse = ", "), ")", 
                         ", sigma_mu_hat=(", paste(round(sqrt(solve(lambda_mu_hat_dd)), 1), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# 予測分布 --------------------------------------------------------------------

# 予測分布のパラメータを計算
lambda_star_hat_dd <- solve(solve(lambda_dd) + solve(lambda_mu_hat_dd))
mu_star_hat_d <- m_hat_d

# 予測分布を計算
predict_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = mu_star_hat_d, sigma = solve(lambda_star_hat_dd)) # 確率密度
) %>% 
  dplyr::select(density) %>% 
  cbind(point_df, .)

# 作図
ggplot() + 
  geom_contour(data = predict_df, aes(x, y, z = density, color = ..level..)) +  # 等高線
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) +  # 真の平均値
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu_star_hat=(", paste(round(mu_star_hat_d, 1), collapse = ", "), ")", 
                         ", sigma_star_hat=(", paste(round(sqrt(solve(lambda_star_hat_dd)), 1), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# gif ---------------------------------------------------------------------

# 追加パッケージ
library(gganimate)


# データ数を指定
N <- 1

# 観測モデルのパラメータを指定
mu_truth_d <- c(25, 50)
sigma_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_dd <- solve(sigma_dd^2)

# 事前分布のパラメータを指定
m_d <- c(0, 0)
sigma_mu_dd <- matrix(c(100, 0, 0, 100), nrow = 2, ncol = 2)
lambda_mu_dd <- solve(sigma_mu_dd^2)


# 作図用の点を生成
x_vec <- seq(mu_truth_d[1] - 2 * sigma_dd[1, 1], mu_truth_d[1] + 2 * sigma_dd[1, 1], by = 0.25)
y_vec <- seq(mu_truth_d[2] - 2 * sigma_dd[2, 2], mu_truth_d[2] + 2 * sigma_dd[2, 2], by = 0.25)
point_df <- tibble(
  x = rep(x_vec, times = length(y_vec)), 
  y = rep(y_vec, each = length(x_vec))
)
mu_df <- tibble(
  x = mu_truth_d[1], 
  y = mu_truth_d[2]
)


# 事前分布を計算
posterior_df <- tidyr::tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = m_d, sigma = solve(lambda_mu_dd)), # 確率密度
  iteration = 0 # 試行回数
) %>% 
  dplyr::select(density, iteration) %>% 
  cbind(point_df, .)


# 予測分布のパラメータを計算
lambda_star_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))
mu_star_d <- m_d

# 初期値による予測分布を計算
predict_df <- tibble(
  xy = point_df, 
  density = mvtnorm::dmvnorm(x = xy, mean = mu_star_d, sigma = solve(lambda_star_dd)), # 確率密度
  iteration = 0 # 試行回数
) %>% 
  dplyr::select(density, iteration) %>% 
  cbind(point_df, .)


# 試行回数を指定
max_iter <- 50

# ベイズ推論
for(i in 1:max_iter) {
  
  # 2次元ガウス分布に従うデータを生成
  x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_truth_d, sigma = solve(lambda_dd))
  
  
  # 事後分布のパラメータを更新
  old_lambda_mu_dd <- lambda_mu_dd
  lambda_mu_dd <- N * lambda_dd + lambda_mu_dd
  m_d <- solve(lambda_mu_dd) %*% (lambda_dd %*% colSums(x_nd) + old_lambda_mu_dd %*% m_d) %>% 
    as.vector()
  
  # 事後分布を計算
  tmp_posterior_df <- tidyr::tibble(
    xy = point_df, 
    density = mvtnorm::dmvnorm(x = xy, mean = m_d, sigma = solve(lambda_mu_dd)), # 確率密度
    iteration = i # 試行回数
  ) %>% 
    dplyr::select(density, iteration) %>% 
    cbind(point_df, .)
  
  
  # 予測分布のパラメータを更新
  lambda_star_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))
  mu_star_d <- m_d
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    xy = point_df, 
    density = mvtnorm::dmvnorm(x = xy, mean = mu_star_d, sigma = solve(lambda_star_dd)), # 確率密度
    iteration = i # 試行回数
  ) %>% 
    dplyr::select(density, iteration) %>% 
    cbind(point_df, .)
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
  
  # 動作確認
  print(i)
}

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_contour(data = posterior_df, aes(x, y, z = density, color = ..level..)) +  # 等高線
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) +  # 平均の点
  transition_manual(iteration) +  # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "i={current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル

# gif画像を作成
animate(posterior_graph, nframes = max_iter + 1, fps = 10)


# 予測分布を作図
predict_graph <- ggplot() + 
  geom_contour(data = predict_df, aes(x, y, z = density, color = ..level..)) +  # 等高線
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) +  # 平均の点
  transition_manual(iteration) +  # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "i={current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル

# gif画像を作成
animate(predict_graph, nframes = max_iter + 1, fps = 10)


