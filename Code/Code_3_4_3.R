
# ch 3.4.3 多次元ガウス分布：平均・精度が未知の場合 ----------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvtnorm)
library(mvnfast)


# モデルの設定 --------------------------------------------------------------------

# データ数を指定
N <- 100
D <- 2 # (固定)

# 観測モデルのパラメータを指定
mu_truth_d <- c(25, 50)
sigma_truth_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_truth_dd <- solve(sigma_truth_dd^2)


# 事前分布のパラメータを指定
m_d <- c(0, 0)
beta <- 1
W_dd <- matrix(c(0.00005, 0, 0, 0.00005), nrow = 2, ncol = 2)
nu <- D


# 作図用の点を生成
x_vec <- seq(mu_truth_d[1] - 2 * sigma_truth_dd[1, 1], mu_truth_d[1] + 2 * sigma_truth_dd[1, 1], by = 0.5)
y_vec <- seq(mu_truth_d[2] - 2 * sigma_truth_dd[2, 2], mu_truth_d[2] + 2 * sigma_truth_dd[2, 2], by = 0.5)
point_df <- tibble(
  x = rep(x_vec, times = length(y_vec)), 
  y = rep(y_vec, each = length(x_vec))
)
mu_df <- tibble(
  x = mu_truth_d[1], 
  y = mu_truth_d[2]
)


# 2次元ガウス分布に従うデータを生成
x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_truth_d, sigma = sigma_truth_dd^2)
summary(x_nd)

# 観測データのデータフレーム
sample_df <- tibble(
  x = x_nd[, 1], 
  y = x_nd[, 2]
)

# 観測モデルのデータフレーム
model_df <- cbind(
  point_df, 
  density = mvnfast::dmvn(
    X = as.matrix(point_df), mu = mu_truth_d, sigma = sigma_truth_dd^2
  ) # 確率密度
)

# 観測データの散布図を作成
ggplot() + 
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..)) + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均パラメータ
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=(", paste(round(mu_truth_d, 1), collapse = ", "), ")", 
                         ", sigma=(", paste(round(sigma_truth_dd, 1), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# 事後分布の計算 -----------------------------------------------------------------

# 事後分布のパラメータを計算
beta_hat <- N + beta
m_hat_d <- (colSums(x_nd) + beta * m_d) / beta_hat
tmp_x <- t(x_nd) %*% as.matrix(x_nd)
tmp_m <- beta * as.matrix(m_d) %*% t(m_d)
tmp_m_hat <- beta_hat * as.matrix(m_hat_d) %*% t(m_hat_d)
W_hat_dd <- solve(
  tmp_x + tmp_m - tmp_m_hat + solve(W_dd)
)
nu_hat <- N + nu
lambda_E_dd <- nu_hat * W_hat_dd


# 事後分布を計算
posterior_df <- cbind(
  point_df, 
  density = mvnfast::dmvn(
    X = as.matrix(point_df), mu = m_hat_d, sigma = solve(lambda_E_dd)
  ) # 確率密度
)

# 作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x, y, z = density, color = ..level..)) + # 事後分布
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均値
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu_hat=(", paste(round(m_hat_d, 1), collapse = ", "), ")", 
                         ", E_sigma_hat=(", paste(round(sqrt(solve(lambda_E_dd)), 1), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# 予測分布の計算 -----------------------------------------------------------------

# 予測分布のパラメータを計算
mu_s_hat_d <- m_hat_d
lambda_s_hat_dd <- (1 - D + nu_hat) * beta_hat / (1 + beta_hat) * W_hat_dd
nu_s_hat <- 1 - D + nu_hat

# 予測分布を計算
predict_df <- cbind(
  point_df, 
  density = mvnfast::dmvt(
    X = as.matrix(point_df), mu = mu_s_hat_d, sigma = solve(lambda_s_hat_dd), df = nu_s_hat
  ) # 確率密度
)

# 作図
ggplot() + 
  geom_contour(data = predict_df, aes(x, y, z = density, color = ..level..)) + # 予測分布
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均パラメータ
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu_s_hat=(", paste(round(mu_s_hat_d, 1), collapse = ", "), ")", 
                         ", lambda_s_hat=(", paste(round(lambda_s_hat_dd, 1), collapse = ", "), ")", 
                         ", df=", nu_s_hat), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル


# gif画像で推移を確認 -------------------------------------------------------------

# 追加パッケージ
library(gganimate)


# データ数を指定
N <- 100
D <- 2 # (固定)

# 観測モデルのパラメータを指定
mu_truth_d <- c(25, 50)
sigma_truth_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_truth_dd <- solve(sigma_truth_dd^2)

# 事前分布のパラメータを指定
m_d <- c(0, 0)
beta <- 1
W_dd <- matrix(c(0.00005, 0, 0, 0.00005), nrow = 2, ncol = 2)
nu <- D
lambda_E_dd <- nu * W_dd

# 作図用の点を生成
x_vec <- seq(mu_truth_d[1] - 3 * sigma_truth_dd[1, 1], mu_truth_d[1] + 3 * sigma_truth_dd[1, 1], by = 1)
y_vec <- seq(mu_truth_d[2] - 3 * sigma_truth_dd[2, 2], mu_truth_d[2] + 3 * sigma_truth_dd[2, 2], by = 1)
point_df <- tibble(
  x = rep(x_vec, times = length(y_vec)), 
  y = rep(y_vec, each = length(x_vec))
)
mu_df <- tibble(
  x = mu_truth_d[1], 
  y = mu_truth_d[2]
)

# 観測モデルを計算
model_df <- cbind(
  point_df, 
  density = mvnfast::dmvn(
    X = as.matrix(point_df), mu = mu_truth_d, sigma = sigma_truth_dd^2
  ) # 確率密度
)


# 事前分布を計算
posterior_df <- cbind(
  point_df, 
  density = mvnfast::dmvn(
    X = as.matrix(point_df), mu = m_d, sigma = solve(lambda_E_dd)
  ), # 確率密度
  iteration = 0 # 試行回数
)


# 予測分布のパラメータを計算
mu_s_d <- m_d
lambda_s_dd <- (1 - D + nu) * beta / (1 + beta) * W_dd
nu_s <- 1 - D + nu

# 予測分布を計算
predict_df <- cbind(
  point_df, 
  density = mvnfast::dmvt(
    X = as.matrix(point_df), mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s
  ), # 確率密度
  iteration = 0 # 試行回数
)


# ベイズ推論
for(n in 1:N) {
  
  # 2次元ガウス分布に従うデータを生成
  x_nd <- mvtnorm::rmvnorm(n = 1, mean = mu_truth_d, sigma = sigma_truth_dd^2)
  
  # 観測データを記録
  if(n > 1) { # 初回以外
    # オブジェクトを結合
    sample_mat <- rbind(sample_mat, x_nd)
    sample_df <- tibble(
      x = sample_mat[, 1],
      y = sample_mat[, 2], 
      iteration = n
    ) %>% 
      rbind(sample_df, .)
  } else if(n == 1){ # 初回
    # オブジェクトを作成
    sample_mat <- x_nd
    sample_df <- tibble(
      x = sample_mat[, 1],
      y = sample_mat[, 2], 
      iteration = n
    )
  }
  
  # 事後分布のパラメータを更新
  old_beta <- beta
  old_m_d <- m_d
  beta <- 1 + beta
  m_d <- as.vector(
    (x_nd + old_beta * m_d) / beta
  )
  tmp_x <- t(x_nd) %*% as.matrix(x_nd)
  tmp_m <- old_beta * as.matrix(old_m_d) %*% t(old_m_d)
  tmp_m_hat <- beta * as.matrix(m_d) %*% t(m_d)
  W_dd <- solve(
    tmp_x + tmp_m - tmp_m_hat + solve(W_dd)
  )
  nu <- 1 + nu
  lambda_E_dd <- nu * W_dd
  
  # 事後分布を計算
  tmp_posterior_df <- cbind(
    point_df, 
    density = mvnfast::dmvn(
      X = as.matrix(point_df), mu = m_d, sigma = solve(lambda_E_dd)
    ), # 確率密度
    iteration = n # 試行回数
  )
  
  # 予測分布のパラメータを計算
  mu_s_d <- m_d
  lambda_s_dd <- (1 - D + nu) * beta / (1 + beta) * W_dd
  nu_s <- 1 - D + nu
  
  # 予測分布を計算
  tmp_predict_df <- cbind(
    point_df, 
    density = mvnfast::dmvt(
      X = as.matrix(point_df), mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s
    ), # 確率密度
    iteration = n # 試行回数
  )
  
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
  
  # 動作確認
  print(n)
}


# 事後分布の期待値を用いた分布を作図
posterior_graph <- ggplot() + 
  geom_contour(data = posterior_df, aes(x, y, z = density, color = ..level..)) + # 精度の期待値を用いた分布
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) + # 平均パラメータ
  transition_manual(iteration) +  # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "N={current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル

# gif画像を作成
animate(posterior_graph, nframes = N + 1, fps = 5)


# 予測分布を作図
predict_graph <- ggplot() + 
  geom_contour(data = predict_df, aes(x, y, z = density, color = ..level..)) + # 予測分布
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  geom_contour(data = model_df, aes(x, y, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 観測モデル
  geom_point(data = mu_df, aes(x = x, y = y), color = "red", shape = 3, size = 5) +  # 平均パラメータ
  transition_manual(iteration) +  # フレーム
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = "N={current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density") # ラベル

# gif画像を作成
animate(predict_graph, nframes = N + 1, fps = 5)


