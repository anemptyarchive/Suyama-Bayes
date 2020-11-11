
# ch3.5 線形回帰の例 ------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvtnorm)


# ch3.5.1 モデルの構築 ----------------------------------------------------------

# (1, x, ..., x^(n-1))のベクトル作成関数を定義
x_vector <- function(x_smp_n, M) {
  x_mn <- matrix(NA, nrow = M, ncol = length(x_smp_n))
  for(m in 1:M) {
    x_mn[m, ] <- x_smp_n^(m-1)
  }
  x_mn
}


# 観測モデルのパラメータを指定
M_truth <- 4
sigma <- 1
lambda <- 1 / sigma^2
w_m <- sample(x = seq(-1, 1, by = 0.1), size = M_truth)


# 作図用のx軸の値
x_line <- seq(-3, 3, by = 0.01)
y_line <- t(w_m) %*% x_vector(x_line, M_truth) %>% 
  as.vector()

# ノイズを含まない観測モデルを作図
model_df <- tibble(
  x = x_line, 
  y = y_line
)
ggplot(model_df, aes(x, y)) + 
  geom_line() + 
  labs(title = "Observation Model", 
       subtitle = paste0("w = (", paste0(round(w_m, 1), collapse = ', '), ")"))


# データをサンプリング
N <- 3
x_smp_n <- sample(seq(min(x_line), max(x_line), by = 0.01), size = N, replace = TRUE)
y_n <- t(w_m) %*% x_vector(x_smp_n, M_truth) + rnorm(n = N, mean = 0, sd = sqrt(1 / lambda))

# 観測データの散布図を作成
sample_df <- tibble(
  x = x_smp_n, 
  y = as.vector(y_n)
)
ggplot() + 
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  geom_line(data = model_df, aes(x, y), color = "blue") + # 誤差なし観測モデル
  labs(title = "Observation Model", 
       subtitle = paste0("M=", M_truth, ", sigma=", round(sqrt(1 / lambda), 1)))


# 事前分布のパラメータを指定
M <- 10
m_m <- matrix(rep(0, M), nrow = M, ncol = 1)
sigma_mm <- diag(M)
lambda_mm <- solve(sigma_mm^2)

# xのベクトルを作成
x_mn <- x_vector(x_smp_n, M)


# 事前分布からのwのサンプリングによるモデルを比較
x_mline <- x_vector(x_line, M)
smp_model_df <- tibble()
for(i in 1:5) {
  tmp_df <- tibble(
    x = x_line, 
    y = as.vector(
      mvtnorm::rmvnorm(n = 1, mean = m_m, sigma = solve(lambda_mm)) %*% x_mline
    ), 
    smp_num = as.factor(i)
  )
  smp_model_df <- rbind(smp_model_df, tmp_df)
}
ggplot(smp_model_df, aes(x, y, color = smp_num)) + 
  ylim(min(min(x_smp_n), min(y_line)), max(max(x_smp_n), max(y_line))) + 
  geom_line()


# 事後分布の計算 -----------------------------------------------------------------

# 事後分布のパラメータを計算
lambda_hat_mm <- lambda * x_mn %*% t(x_mn) + lambda_mm
m_hat_m <- solve(lambda_hat_mm) %*% (lambda * x_mn %*% t(y_n) + lambda_mm %*% m_m)


# 事後分布による観測モデルのパラメータのサンプリングによるモデルの変化を比較
x_mline <- x_vector(x_line, M)
smp_model_df <- tibble()
for(i in 1:5) {
  tmp_df <- tibble(
    x = x_line, 
    y = as.vector(
      mvtnorm::rmvnorm(n = 1, mean = m_hat_m, sigma = solve(lambda_hat_mm)) %*% x_mline
    ), 
    smp_num = as.factor(i)
  )
  smp_model_df <- rbind(smp_model_df, tmp_df)
}
ggplot() + 
  geom_line(data = smp_model_df, aes(x, y, color = smp_num)) + # 事前分布からサンプリングしたモデル
  geom_line(data = model_df, aes(x, y), color = "blue", linetype = "dotted") + # 観測モデル
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  ylim(min(min(x_smp_n), min(y_line)), max(max(x_smp_n), max(y_line))) + 
  labs(title = "Sampling", 
       subtitle = paste0("M=", M))
  


# 予測分布の計算 -----------------------------------------------------------------


# 予測分布のパラメータを計算
sigma2_star_line <- rep(NA, length(x_line))
for(i in seq_along(x_line)) {
  sigma2_star_line[i] <- 1 / lambda + t(x_mline[, i]) %*% solve(lambda_hat_mm) %*% matrix(x_mline[, i])
}
mu_star_line <- t(m_hat_m) %*% x_mline %>% 
  as.vector()

# 予測分布を計算
predict_df <- tibble(
  x = x_line, 
  E_y = mu_star_line, 
  minus_sigma_y = E_y - sqrt(sigma2_star_line), 
  plus_sigma_y = E_y + sqrt(sigma2_star_line)
)

# 作図
ggplot() + 
  geom_line(data = predict_df, aes(x, E_y), color = "orange") + # 予測分布の期待値
  geom_line(data = predict_df, aes(x, plus_sigma_y), color = "#00A968", linetype = "dashed") + # 予測分布の期待値+sigma
  geom_line(data = predict_df, aes(x, minus_sigma_y), color = "#00A968", linetype = "dashed") + # 予測分布の期待値-sigma
  geom_line(data = model_df, aes(x, y), color = "blue", linetype = "dotted") + # 観測モデル
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  ylim(min(min(x_smp_n), min(y_line)), max(max(x_smp_n), max(y_line))) + 
  labs(title = "Predictive Distribution", 
       subtitle = paste0("M=", M), 
       y = "y")



print(summary(sigma2_star_line))



# gif画像でサンプルサイズによる分布の変化を確認 ------------------------------------------------

# 追加パッケージ
library(gganimate)


# (1, x, ..., x^(n-1))のベクトル作成関数を定義
x_vector <- function(x_smp_n, M) {
  x_mn <- matrix(NA, nrow = M, ncol = length(x_smp_n))
  for(m in 1:M) {
    x_mn[m, ] <- x_smp_n^(m-1)
  }
  x_mn
}


# 観測モデルのパラメータを指定
M_truth <- 4
sigma <- 1
lambda <- 1 / sigma^2
w_m <- sample(x = seq(-1, 1, by = 0.1), size = M_truth) %>% 
  matrix()


# 作図用のx軸の値
x_line <- seq(-3, 3, by = 0.01)
y_line <- t(w_m) %*% x_vector(x_line, M_truth) %>% 
  as.vector()

# ノイズを含まない観測モデルを作図
model_df <- tibble(
  x = x_line, 
  y = y_line
)
ggplot(model_df, aes(x, y)) + 
  geom_line() +
  labs(title = "Observation Model", 
       subtitle = paste0("w = (", paste0(round(w_m, 1), collapse = ', '), ")"))


# サンプルサイズを指定
N <- 100

# 事前分布のパラメータを指定
M <- 10
m_m <- matrix(rep(0, M), nrow = M, ncol = 1)
sigma_mm <- diag(M)
lambda_mm <- solve(sigma_mm^2)


# 推論
x_smp_n <- rep(NA, N)
y_n <- rep(NA, N)
eps_n <- rep(NA, N)
x_mline <- x_vector(x_line, M)
sample_df <- tibble()
predict_df <- tibble()
for(n in 1:N) {
  
  # データをサンプリング
  x_smp_n[n] <- sample(seq(min(x_line), max(x_line), by = 0.01), size = 1, replace = TRUE)
  y_n[n] <- t(w_m) %*% x_vector(x_smp_n[n], M_truth) + rnorm(n = 1, mean = 0, sd = sqrt(1 / lambda))
  
  # 観測データのデータフレームを作成
  sample_df <- tibble(
    x = x_smp_n[1:n], 
    y = as.vector(y_n[1:n]), 
    iteration = n
  ) %>% 
    rbind(sample_df, .)
  
  # 観測データからxベクトルを作成
  x_mn <- x_vector(x_smp_n[n], M)
  
  # 事後分布のパラメータを計算
  old_lambda_mm <- lambda_mm
  lambda_mm <- lambda * x_mn %*% t(x_mn) + lambda_mm
  m_m <- solve(lambda_mm) %*% (lambda * x_mn %*% t(y_n[n]) + old_lambda_mm %*% m_m)
  
  # 予測分布のパラメータを計算
  sigma2_star_line <- rep(NA, length(x_line))
  for(i in seq_along(x_line)) {
    sigma2_star_line[i] <- 1 / lambda + t(x_mline[, i]) %*% solve(lambda_mm) %*% matrix(x_mline[, i])
  }
  mu_star_line <- t(m_m) %*% x_mline %>% 
    as.vector()
  
  # 予測分布を計算
  predict_df <- tibble(
    x = x_line, 
    E_y = mu_star_line, 
    minus_sigma_y = E_y - sqrt(sigma2_star_line), 
    plus_sigma_y = E_y + sqrt(sigma2_star_line), 
    iteration = n
  ) %>% 
    rbind(predict_df, .)
}


# 作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x, E_y), color = "orange") + # 予測分布の期待値
  geom_line(data = predict_df, aes(x, plus_sigma_y), color = "#00A968", linetype = "dashed") + # 予測分布の期待値+sigam
  geom_line(data = predict_df, aes(x, minus_sigma_y), color = "#00A968", linetype = "dashed") + # 予測分布の期待値-sigma
  geom_line(data = model_df, aes(x, y), color = "blue", linetype = "dotted") + # 観測モデル
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  transition_manual(iteration) + # フレーム
  ylim(min(min(x_smp_n), min(y_line)), max(max(x_smp_n), max(y_line))) + 
  labs(title = "Predictive Distribution", 
       subtitle = paste("M=", M, ", N={current_frame}", sep = ""), 
       y = "y")

# gif画像を作成
animate(predict_graph, nframes = N, fps = 10)

warnings()


# gif画像で次元数による分布の変化を確認 ------------------------------------------------

# 追加パッケージ
library(gganimate)


# (1, x, ..., x^(n-1))のベクトル作成関数を定義
x_vector <- function(x_smp_n, M) {
  x_mn <- matrix(NA, nrow = M, ncol = length(x_smp_n))
  for(m in 1:M) {
    x_mn[m, ] <- x_smp_n^(m-1)
  }
  x_mn
}


# 観測モデルのパラメータを指定
M_truth <- 4
sigma <- 1
lambda <- 1 / sigma^2
w_m <- sample(x = seq(-1, 1, by = 0.1), size = M_truth) %>% 
  matrix()


# 作図用のx軸の値
x_line <- seq(-3, 3, by = 0.01)
y_line <- t(w_m) %*% x_vector(x_line, M_truth) %>% 
  as.vector()


# サンプルサイズを指定
N <- 10

# データをサンプリング
x_smp_n <- sample(seq(min(x_line), max(x_line), by = 0.01), size = N, replace = TRUE)
y_n <- t(w_m) %*% x_vector(x_smp_n, M_truth) + rnorm(n = N, mean = 0, sd = sqrt(1 / lambda)) 


# 観測データのデータフレームを作成
model_df <- tibble(
  x = x_line, 
  y = y_line
)
sample_df <- tibble(
  x = x_smp_n, 
  y = as.vector(y_n)
)
ggplot() + 
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  geom_line(data = model_df, aes(x, y), color = "blue") + # 誤差なし観測モデル
  labs(title = "Observation Model", 
       subtitle = paste0("M=", M_truth, ", sigma=", round(sqrt(1 / lambda), 1)))


# 事前分布のパラメータを指定
max_M <- 15


# 推論
predict_df <- tibble()
for(m in 1:max_M) {
  
  # 事前分布のパラメータを生成
  m_m <- matrix(rep(0, m), nrow = m, ncol = 1)
  sigma_mm <- diag(m)
  lambda_mm <- solve(sigma_mm^2)
  
  # 観測データからxベクトルを作成
  x_mn <- x_vector(x_smp_n, m)
  x_mline <- x_vector(x_line, m)
  
  # 事後分布のパラメータを計算
  old_lambda_mm <- lambda_mm
  lambda_mm <- lambda * x_mn %*% t(x_mn) + lambda_mm
  m_m <- solve(lambda_mm) %*% (lambda * x_mn %*% t(y_n) + old_lambda_mm %*% m_m)
  
  # 予測分布のパラメータを計算
  sigma2_star_line <- rep(NA, length(x_line))
  for(i in seq_along(x_line)) {
    sigma2_star_line[i] <- 1 / lambda + t(x_mline[, i]) %*% solve(lambda_mm) %*% matrix(x_mline[, i])
  }
  mu_star_line <- t(m_m) %*% x_mline %>% 
    as.vector()
  
  # 予測分布を計算
  predict_df <- tibble(
    x = x_line, 
    E_y = mu_star_line, 
    minus_sigma_y = E_y - sqrt(sigma2_star_line), 
    plus_sigma_y = E_y + sqrt(sigma2_star_line), 
    iteration = m
  ) %>% 
    rbind(predict_df, .)
}


# 作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x, E_y), color = "orange") + # 予測分布の期待値
  geom_line(data = predict_df, aes(x, plus_sigma_y), color = "#00A968", linetype = "dashed") + # 予測分布の期待値+sigam
  geom_line(data = predict_df, aes(x, minus_sigma_y), color = "#00A968", linetype = "dashed") + # 予測分布の期待値-sigma
  geom_line(data = model_df, aes(x, y), color = "blue", linetype = "dotted") + # 観測モデル
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  transition_manual(iteration) + # フレーム
  ylim(min(min(x_smp_n), min(y_line)), max(max(x_smp_n), max(y_line))) + 
  labs(title = "Predictive Distribution", 
       subtitle = paste0("M={current_frame}", ", N=", N), 
       y = "y")

# gif画像を作成
animate(predict_graph, nframes = max_M)

warnings()

