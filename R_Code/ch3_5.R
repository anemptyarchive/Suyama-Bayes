
# ch3.5 線形回帰の例 ------------------------------------------------------------

# 3.5節で利用パッケージ
library(tidyverse)
library(mvtnorm)


### モデルの構築 -----

# ベクトル(1, x^1, ..., x^(m-1))の作成関数を定義
x_vector <- function(x_n, M) {
  # 受け皿を作成
  x_nm <- matrix(NA, nrow = length(x_n), ncol = M)
  
  # m乗を計算
  for(m in 1:M) {
    x_nm[, m] <- x_n^(m-1)
  }
  return(x_nm)
}


# 真の次元数を指定
M_truth <- 4

# ノイズ成分の標準偏差を指定
sigma <- 1.5

# ノイズ成分の精度を計算
lambda <- 1 / sigma^2
lambda

# パラメータを生成
w_truth_m <- sample(x = seq(-1, 1, by = 0.1), size = M_truth)
w_truth_m


# 作図用のx軸の値を作成
x_line <- seq(-3, 3, by = 0.01)

# 作図用のxをM次元に拡張
x_truth_mat <- x_vector(x_line, M_truth)

# 真の観測モデルをデータフレームに格納
model_df <- tibble(
  x = x_line, 
  y = as.vector(t(w_truth_m) %*% t(x_truth_mat))
)

# 真の観測モデルを作図
ggplot(model_df, aes(x = x, y = y)) + 
  geom_line() + 
  labs(title = "Observation Model", 
       subtitle = paste0("w=(", paste0(round(w_truth_m, 1), collapse = ', '), ")"))


## 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# 入力値を生成
x_n <- sample(seq(min(x_line), max(x_line), by = 0.01), size = N, replace = TRUE)

# 入力値をM次元に拡張
x_truth_nm <- x_vector(x_n, M_truth)

# ノイズ成分を生成
epsilon_n <- rnorm(n = N, mean = 0, sd = sqrt(1 / lambda))

# 出力値を計算
y_n <- (t(w_truth_m) %*% t(x_truth_nm) + epsilon_n) %>% 
  as.vector()
y_n

# 観測データをデータフレームに格納
sample_df <- tibble(
  x = x_n, 
  y = y_n
)

# 観測データの散布図を作成
ggplot() + 
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  geom_line(data = model_df, aes(x, y), color = "blue") + # 真のモデル
  labs(title = "Observation Data", 
       subtitle = paste0("N=", N, 
                         ", w=(", paste0(round(w_truth_m, 1), collapse = ', '), ")", 
                         ", sigma=", round(sqrt(1 / lambda), 1)))


### 事前分布の設定 -----

# 事前分布の次元数を指定
M <- 5

# 事前分布の平均を指定
m_m <- rep(0, M)

# 事前分布の精度行列を指定
sigma_mm <- diag(M) * 100
lambda_mm <- solve(sigma_mm)
lambda_mm


# 入力値をM次元に拡張
x_nm <- x_vector(x_n, M)

# 作図用のxをM次元に拡張
x_mat <- x_vector(x_line, M)

# 事前分布からのwのサンプリング
prior_df <- tibble()
for(i in 1:5) { # サンプルサイズを指定
  # パラメータを生成
  prior_w_m <- mvtnorm::rmvnorm(n = 1, mean = m_m, sigma = solve(lambda_mm)) %>% 
    as.vector()
  
  # 入出力値をデータフレームに格納
  tmp_df <- tibble(
    x = x_line, 
    y = as.vector(t(prior_w_m) %*% t(x_mat)), 
    smp_num = as.factor(i) # サンプル番号
  )
  
  # 結果を結合
  prior_df <- rbind(prior_df, tmp_df)
}

# 事前分布からサンプリングしたパラメータによる分布を作図
ggplot() + 
  geom_line(data = prior_df, aes(x = x, y = y, color = smp_num)) + # サンプリングしたwを用いたモデル
  geom_line(data = model_df, aes(x =x, y = y), color = "blue", linetype = "dashed") + # 真のモデル
  ylim(min(model_df[["y"]]), max(model_df[["y"]])) + # y軸の表示範囲
  labs(title = "Sampling from Prior Distribution", 
       subtitle = paste0("m=(", paste0(m_m, collapse = ", "), ")"))


### 事後分布の計算 -----

# 事後分布のパラメータを計算:式(3.148)
lambda_hat_mm <- lambda * t(x_nm) %*% x_nm + lambda_mm
m_hat_m <- solve(lambda_hat_mm) %*% (lambda * t(t(y_n) %*% x_nm) + lambda_mm %*% matrix(m_m))


# 事後分布からのwのサンプリングによるモデルを比較
posterior_df <- tibble()
for(i in 1:5) { # サンプルサイズを指定
  # パラメータを生成
  posterior_w_m <- mvtnorm::rmvnorm(n = 1, mean = m_hat_m, sigma = solve(lambda_hat_mm)) %>% 
    as.vector()
  
  # 入出力値をデータフレームに格納
  tmp_df <- tibble(
    x = x_line, 
    y = as.vector(t(posterior_w_m) %*% t(x_mat)), 
    smp_num = as.factor(i) # サンプル番号
  )
  
  # 結果を結合
  posterior_df <- rbind(posterior_df, tmp_df)
}

# 事後分布からサンプリングしたパラメータによる分布を作図
ggplot() + 
  geom_line(data = posterior_df, aes(x = x, y = y, color = smp_num)) + # サンプリングしたwを用いたモデル
  geom_line(data = model_df, aes(x = x, y = y), color = "blue", linetype = "dashed") + # 真のモデル
  geom_point(data = sample_df, aes(x, y)) + # 観測データ
  #ylim(min(model_df[["y"]]), max(model_df[["y"]])) + # y軸の表示範囲
  labs(title = "Sampling from Posterior Distribution", 
       subtitle = paste0("N=", N, ", m=(", paste0(round(m_hat_m, 2), collapse = ", "), ")"))
  

### 予測分布の計算 -----

# 予測分布のパラメータを計算:式(3.155')
mu_star_hat_line <- t(m_hat_m) %*% t(x_mat) %>% 
  as.vector()
sigma2_star_hat_line <- 1 / lambda + x_mat %*% solve(lambda_hat_mm) %*% t(x_mat) %>% 
  diag()


# 予測分布をデータフレームに格納
predict_df <- tibble(
  x = x_line, 
  E_y = mu_star_hat_line, # 予測分布の期待値
  minus_sigma_y = E_y - sqrt(sigma2_star_hat_line), # 予測分布の期待値-sigma
  plus_sigma_y = E_y + sqrt(sigma2_star_hat_line) # 予測分布の期待値+sigma
)

# 予測分布を作図
ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = E_y), color = "orange") + # 予測分布の期待値
  geom_ribbon(data = predict_df, aes(x = x, ymin = minus_sigma_y, ymax = plus_sigma_y), 
              fill = "#00A968", alpha = 0.1, color = "#00A968", linetype = "dotted") + # 予測分布の標準偏差
  geom_line(data = model_df, aes(x = x, y = y), color = "blue", linetype = "dashed") + # 真のモデル
  geom_point(data = sample_df, aes(x = x, y = y)) + # 観測データ
  ylim(min(model_df[["y"]]) - 3 * sigma, max(model_df[["y"]]) + 3 * sigma) + # y軸の表示範囲
  labs(title = "Predictive Distribution", 
       subtitle = paste0("N=", N, ", m=(", paste0(round(m_hat_m, 2), collapse = ", "), ")"), 
       y = "y")


# ・サンプルサイズによる分布の変化をアニメーションで確認 ------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(mvtnorm)
library(gganimate)


### モデルの構築 -----

# ベクトル(1, x^1, ..., x^(m-1))の作成関数を定義
x_vector <- function(x_n, M) {
  # 受け皿を作成
  x_nm <- matrix(NA, nrow = length(x_n), ncol = M)
  
  # m乗を計算
  for(m in 1:M) {
    x_nm[, m] <- x_n^(m-1)
  }
  return(x_nm)
}

# 真の次元数を指定
M_truth <- 4

# ノイズ成分の標準偏差を指定
sigma <- 1.5

# ノイズ成分の精度を計算
lambda <- 1 / sigma^2
lambda

# パラメータを生成
w_truth_m <- sample(x = seq(-1, 1, by = 0.1), size = M_truth)
w_truth_m


# 作図用のx軸の値を作成
x_line <- seq(-3, 3, by = 0.01)

# 観測モデルをデータフレームに格納
model_df <- tibble(
  x = x_line, 
  y = t(w_truth_m) %*% t(x_vector(x_line, M_truth)) %>% 
    as.vector()
)

# 観測モデルを作図
ggplot(model_df, aes(x = x, y = y)) + 
  geom_line() + 
  labs(title = "Observation Model", 
       subtitle = paste0("w=(", paste0(round(w_truth_m, 1), collapse = ', '), ")"))


### 推論処理 -----

# (観測)データ数(試行回数)を指定
N <- 100

# 事前分布の次元数を指定
M <- 5

# 事前分布の平均を指定
m_m <- rep(0, M)

# 事前分布の精度行列を指定
lambda_mm <- diag(M) * 0.01
solve(lambda_mm)


# 作図用のxをM次元に拡張
x_mat <- x_vector(x_line, M)

# 初期値による予測分布のパラメータを計算:式(3.155)
mu_star_line <- t(m_m) %*% t(x_mat) %>% 
  as.vector()
sigma2_star_line <- 1 / lambda + x_mat %*% solve(lambda_mm) %*% t(x_mat) %>% 
  diag()

# 初期値による予測分布をデータフレームに格納
predict_df <- tibble(
  x = x_line, 
  E_y = mu_star_line, 
  minus_sigma_y = E_y - sqrt(sigma2_star_line), 
  plus_sigma_y = E_y + sqrt(sigma2_star_line), 
  label = as.factor(
    paste0(
      "N=", 0, ", m=(", paste(m_m, collapse = ", "), ")"
    )
  )
)


# 受け皿を初期化
x_n <- rep(NA, N)
y_n <- rep(NA, N)
x_df <- tibble(
  x = NA, 
  y = NA, 
  label = as.factor(
    paste0("N=", 0, ", m=(", paste(m_m, collapse = ", "), ")")
  )
)

# ベイズ推論
for(n in 1:N) {
  
  # 入力値を生成
  x_n[n] <- sample(seq(min(x_line), max(x_line), by = 0.01), size = 1, replace = TRUE)
  
  # 出力値を計算:式(3.141)
  y_n[n] <- t(w_truth_m) %*% t(x_vector(x_n[n], M_truth)) + rnorm(n = 1, mean = 0, sd = sqrt(1 / lambda))
  
  # 入力値をM次元に拡張
  x_1m <- x_vector(x_n[n], M)
  
  # 事後分布のパラメータを計算:式(3.148)
  old_lambda_mm <- lambda_mm
  lambda_mm <- lambda * t(x_1m) %*% x_1m + lambda_mm
  m_m <- solve(lambda_mm) %*% (lambda * y_n[n] * t(x_1m) + old_lambda_mm %*% matrix(m_m))
  
  # 予測分布のパラメータを計算:式(3.155')
  mu_star_line <- t(m_m) %*% t(x_mat) %>% 
    as.vector()
  sigma2_star_line <- 1 / lambda + x_mat %*% solve(lambda_mm) %*% t(x_mat) %>% 
    diag()
  
  # n回目までの観測データをデータフレームに格納
  tmp_x_df <- tibble(
    x = x_n[1:n], 
    y = as.vector(y_n[1:n]), 
    label = as.factor(
      paste0(
        "N=", n, ", m=(", paste(round(m_m, 2), collapse = ", "), ")"
      )
    )
  )
  x_df <- rbind(x_df, tmp_x_df)
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = x_line, 
    E_y = mu_star_line, 
    minus_sigma_y = E_y - sqrt(sigma2_star_line), 
    plus_sigma_y = E_y + sqrt(sigma2_star_line), 
    label = as.factor(
      paste0(
        "N=", n, ", m=(", paste(round(m_m, 2), collapse = ", "), ")"
      )
    )
  )
  predict_df <- rbind(predict_df, tmp_predict_df)
}


### 作図処理 -----

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = E_y), color = "orange") + # 予測分布の期待値
  geom_ribbon(data = predict_df, aes(x = x, ymin = minus_sigma_y, ymax = plus_sigma_y), 
              fill = "#00A968", alpha = 0.1, color = "#00A968", linetype = "dotted") + # 予測分布の標準偏差
  geom_line(data = model_df, aes(x = x, y = y), color = "blue", linetype = "dashed") + # 真のモデル
  geom_point(data = x_df, aes(x = x, y = y)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  ylim(min(model_df[["y"]]) - 3 * sigma, max(model_df[["y"]]) + 3 * sigma) + # y軸の表示範囲
  labs(title = "Predictive Distribution", 
       subtitle = "{current_frame}", 
       y = "y")

# gif画像を作成
gganimate::animate(predict_graph, nframes = N, fps = 10)

warnings()


# ・次元数による分布の変化をアニメーションで確認 ------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(mvtnorm)
library(gganimate)


### モデルの構築 -----

# ベクトル(1, x^1, ..., x^(m-1))の作成関数を定義
x_vector <- function(x_n, M) {
  # 受け皿を作成
  x_nm <- matrix(NA, nrow = length(x_n), ncol = M)
  
  # m乗を計算
  for(m in 1:M) {
    x_nm[, m] <- x_n^(m-1)
  }
  return(x_nm)
}

# 真の次元数を指定
M_truth <- 4

# ノイズ成分の標準偏差を指定
sigma <- 1.5

# ノイズ成分の精度を計算
lambda <- 1 / sigma^2
lambda

# パラメータを生成
w_truth_m <- sample(x = seq(-1, 1, by = 0.1), size = M_truth)
w_truth_m


# 作図用のx軸の値を作成
x_line <- seq(-3, 3, by = 0.01)

# 観測モデルをデータフレームに格納
model_df <- tibble(
  x = x_line, 
  y = t(w_truth_m) %*% t(x_vector(x_line, M_truth)) %>% 
    as.vector()
)

# 観測モデルを作図
ggplot(model_df, aes(x = x, y = y)) + 
  geom_line() + 
  labs(title = "Observation Model", 
       subtitle = paste0("w=(", paste0(round(w_truth_m, 1), collapse = ', '), ")"))


### 観測データの生成 -----

# (観測)データ数(試行回数)を指定
N <- 5

# 入力値を生成
x_n <- sample(seq(min(x_line), max(x_line), by = 0.01), size = N, replace = TRUE)

# 出力値を計算:式(3.141)
y_n <- (t(w_truth_m) %*% t(x_vector(x_n, M_truth)) + rnorm(n = N, mean = 0, sd = sqrt(1 / lambda))) %>% 
  as.vector()
y_n

# 観測データをデータフレームに格納
x_df <- tibble(
  x = x_n, 
  y = y_n
)

# 観測データの散布図を作成
ggplot() + 
  geom_point(data = x_df, aes(x = x, y = y)) + # 観測データ
  geom_line(data = model_df, aes(x = x, y = y), color = "blue") + # 真のモデル
  labs(title = "Observation Data", 
       subtitle = paste0("N=", N, ", w=(", paste0(round(w_truth_m, 1), collapse = ', '), ")", 
                         ", sigma=", round(sqrt(1 / lambda), 1)))


### 推論処理 -----

# 事前分布の次元数の最大値(試行回数)を指定
M_max <- 15

# 受け皿を作成
predict_df <- tibble()

# ベイズ推論
for(m in 1:M_max) {
  
  # 事前分布のパラメータをm次元に初期化
  m_m <- rep(0, m)
  lambda_mm <- diag(m) * 0.01
  
  # 入力値をm次元に拡張
  x_nm <- x_vector(x_n, m)
  
  # 作図用のxをm次元に拡張
  x_mat <- x_vector(x_line, m)
  
  # 事後分布のパラメータを計算:式(3.148)
  lambda_hat_mm <- lambda * t(x_nm) %*% x_nm + lambda_mm
  m_hat_m <- solve(lambda_hat_mm) %*% (lambda * t(t(y_n) %*% x_nm) + lambda_mm %*% matrix(m_m)) %>% 
    as.vector()
  
  # 予測分布のパラメータを計算:式(3.155)
  mu_star_hat_line <- t(m_hat_m) %*% t(x_mat) %>% 
    as.vector()
  sigma2_star_hat_line <- 1 / lambda + x_mat %*% solve(lambda_hat_mm) %*% t(x_mat) %>% 
    diag()
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = x_line, 
    E_y = mu_star_hat_line, 
    minus_sigma_y = E_y - sqrt(sigma2_star_hat_line), 
    plus_sigma_y = E_y + sqrt(sigma2_star_hat_line), 
    label = as.factor(
      paste0(
        "N=", N, ", M=", m, ", m=(", paste(round(m_hat_m, 2), collapse = ", "), ")"
      )
    )
  )
  predict_df <- rbind(predict_df, tmp_predict_df)
}


### 作図処理 -----

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = E_y), color = "orange") + # 予測分布の期待値
  geom_ribbon(data = predict_df, aes(x = x, ymin = minus_sigma_y, ymax = plus_sigma_y), 
              fill = "#00A968", alpha = 0.1, color = "#00A968", linetype = "dotted") + # 予測分布の標準偏差
  geom_line(data = model_df, aes(x = x, y = y), color = "blue", linetype = "dashed") + # 真のモデル
  geom_point(data = x_df, aes(x = x, y = y)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  ylim(min(model_df[["y"]]) - 3 * sigma, max(model_df[["y"]]) + 3 * sigma) + # y軸の表示範囲
  labs(title = "Predictive Distribution", 
       subtitle = "{current_frame}", 
       y = "y")

# gif画像を作成
gganimate::animate(predict_graph, nframes = M_max, fps = 10)

warnings()


