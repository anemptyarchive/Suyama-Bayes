
# 3.4.1 多次元ガウス分布：平均が未知の場合 ----------------------------------------------

# 3.4.1項で利用パッケージ
library(tidyverse)
library(mvnfast)


### 尤度(多次元ガウス分布)の設定 -----

# 真のパラメータを指定
mu_truth_d <- c(25, 50)
sigma_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_dd <- solve(sigma_dd^2)

# 作図用のxの点を作成
x_1_vec <- seq(mu_truth_d[1] - 4 * sigma_dd[1, 1], mu_truth_d[1] + 4 * sigma_dd[1, 1], length.out = 1000)
x_2_vec <- seq(mu_truth_d[2] - 4 * sigma_dd[2, 2], mu_truth_d[2] + 4 * sigma_dd[2, 2], length.out = 1000)
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)

# 尤度を計算:式(2.72)
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_truth_d, sigma = solve(lambda_dd)
  )
)

# 尤度を作図
ggplot(model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + 
  geom_contour() + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
     subtitle = paste0("mu=(", paste(round(mu_truth_d, 1), collapse = ", "), ")", 
                       ", lambda=(", paste(round(lambda_dd, 5), collapse = ", "), ")"), 
     x = expression(x[1]), y = expression(x[2]), 
     color = "density")


### 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_truth_d, sigma = solve(lambda_dd))
#x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_truth_d, sigma = solve(lambda_dd))

# 観測データを確認
summary(x_nd)

# 観測データのデータフレームを作成
x_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2]
)

# 観測データの散布図を作図
ggplot() + 
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=(", paste(mu_truth_d, collapse = ", "), ")", 
                         ", sigma=(", paste(sqrt(solve(lambda_dd)), collapse = ", "), ")"), 
      x = expression(x[1]), y = expression(x[2]), 
      color = "density")


### 事前分布(多次元ガウス分布)の設定 -----

# muの事前分布のパラメータを指定
m_d <- c(0, 0)
sigma_mu_dd <- matrix(c(100, 0, 0, 100), nrow = 2, ncol = 2)
lambda_mu_dd <- solve(sigma_mu_dd^2)

# 作図用のmuの点を作成
mu_1_vec <- seq(mu_truth_d[1] - 100, mu_truth_d[1] + 100, length.out = 1000)
mu_2_vec <- seq(mu_truth_d[2] - 100, mu_truth_d[2] + 100, length.out = 1000)
mu_point_mat <- cbind(
  rep(mu_1_vec, times = length(mu_2_vec)), 
  rep(mu_2_vec, each = length(mu_1_vec))
)

# muの事前分布を計算:式(2.72)
prior_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = mu_point_mat, mu = m_d, sigma = solve(lambda_mu_dd)
  )
)

# 作図用に真のmuのデータフレームを作成
mu_df <- tibble(
  mu_1 = mu_truth_d[1], 
  mu_2 = mu_truth_d[2]
)

# muの事前分布を作図
ggplot() + 
  geom_contour(data = prior_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # muの事前分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("m=(", paste(round(m_d, 1), collapse = ", "), ")", 
                         ", lambda_mu=(", paste(round(lambda_mu_dd, 5), collapse = ", "), ")"), 
       x = expression(mu[1]), y = expression(mu[2]), 
       color = "density")


### 事後分布(多次元ガウス分布)の計算 -----

# muの事後分布のパラメータを計算:式(3.102),(3.103)
lambda_mu_hat_dd <- N * lambda_dd + lambda_mu_dd
m_hat_d <- solve(lambda_mu_hat_dd) %*% (lambda_dd %*% colSums(x_nd) + lambda_mu_dd %*% m_d) %>% 
  as.vector()

# muの事後分布を計算:式(2.72)
posterior_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = , mu_point_mat, mu = m_hat_d, sigma = solve(lambda_mu_hat_dd)
  )
)

# muの事後分布を作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # muの事後分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  xlim(c(min(mu_point_mat[, 1]), max(mu_point_mat[, 1]))) + # x軸の表示範囲
  ylim(c(min(mu_point_mat[, 2]), max(mu_point_mat[, 2]))) + # y軸の表示範囲
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=(", paste(round(m_hat_d, 1), collapse = ", "), ")", 
                         ", sigma_mu_hat=(", paste(round(sqrt(solve(lambda_mu_hat_dd)), 1), collapse = ", "), ")"), 
       x = expression(mu[1]), y = expression(mu[2]), 
       color = "density")


### 予測分布(多次元ガウス分布)の計算 -----

# 予測分布のパラメータを計算:式(3.109'),(3.110')
lambda_star_hat_dd <- solve(solve(lambda_dd) + solve(lambda_mu_hat_dd))
mu_star_hat_d <- m_hat_d

# 予測分布を計算:式(2.72)
predict_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_star_hat_d, sigma = solve(lambda_star_hat_dd)
  )
)

# 予測分布を作図
ggplot() + 
  geom_contour(data = predict_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 予測分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu_star_hat=(", paste(round(mu_star_hat_d, 1), collapse = ", "), ")", 
                         ", lambda_star_hat=(", paste(round(lambda_star_hat_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


# ・アニメーション ---------------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)


### 推論処理 -----

# 真のパラメータを指定
mu_truth_d <- c(25, 50)
sigma_dd <- matrix(c(50, 30, 30, 50), nrow = 2, ncol = 2)
lambda_dd <- solve(sigma_dd^2)

# muの事前分布のパラメータを指定
m_d <- c(0, 0)
sigma_mu_dd <- matrix(c(100, 0, 0, 100), nrow = 2, ncol = 2)
lambda_mu_dd <- solve(sigma_mu_dd^2)

# 初期値による予測分布のパラメータを計算:式(3.109),(3.110)
lambda_star_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))
mu_star_d <- m_d


# 作図用のmuの点を作成
mu_1_vec <- seq(mu_truth_d[1] - 100, mu_truth_d[1] + 100, length.out = 500)
mu_2_vec <- seq(mu_truth_d[2] - 100, mu_truth_d[2] + 100, length.out = 500)
mu_point_mat <- cbind(
  rep(mu_1_vec, times = length(mu_2_vec)), 
  rep(mu_2_vec, each = length(mu_1_vec))
)

# muの事前分布を計算:式(2.72)
posterior_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = mu_point_mat, mu = m_d, sigma = solve(lambda_mu_dd)
  ), 
  label = as.factor(
    paste0(
      "N=", 0, 
      ", m=(", paste(round(m_d, 1), collapse = ", "), ")", 
      ", lambda_mu=(", paste(round(lambda_mu_dd, 5), collapse = ", "), ")"
    )
  )
)


# 作図用のxの点を作成
x_1_vec <- seq(mu_truth_d[1] - 4 * sigma_dd[1, 1], mu_truth_d[1] + 4 * sigma_dd[1, 1], length.out = 500)
x_2_vec <- seq(mu_truth_d[2] - 4 * sigma_dd[2, 2], mu_truth_d[2] + 4 * sigma_dd[2, 2], length.out = 500)
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)

# 予測分布を計算:式(2.72)
predict_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_star_d, sigma = solve(lambda_star_dd)
  ), 
  label = as.factor(
    paste0(
      "N=", 0, 
      ", mu_star=(", paste(round(mu_star_d, 1), collapse = ", "), ")", 
      ", lambda_star=(", paste(round(lambda_star_dd, 5), collapse = ", "), ")"
    )
  )
)


# データ数(試行回数)を指定
N <- 50

# 観測データの受け皿を初期化
x_nd <- matrix(0, nrow = N, ncol = 2)

# ベイズ推論
for(n in 1:N) {
  
  # 多次元ガウス分布に従うデータを生成
  x_nd[n, ] <- mvnfast::rmvn(n = 1, mu = mu_truth_d, sigma = solve(lambda_dd)) %>% 
    as.vector()
  
  # 事後分布のパラメータを更新:式(3.102),(3.103)
  old_lambda_mu_dd <- lambda_mu_dd
  lambda_mu_dd <- lambda_dd + lambda_mu_dd
  m_d <- solve(lambda_mu_dd) %*% (lambda_dd %*% x_nd[n, ] + old_lambda_mu_dd %*% m_d) %>% 
    as.vector()
  
  # muの事後分布を計算:式(2.72)
  tmp_posterior_df <- tibble(
    mu_1 = mu_point_mat[, 1], 
    mu_2 = mu_point_mat[, 2], 
    density = mvnfast::dmvn(
      X = mu_point_mat, mu = m_d, sigma = solve(lambda_mu_dd)
    ), 
    label = as.factor(
      paste0(
        "N=", n, 
        ", m_hat=(", paste(round(m_d, 1), collapse = ", "), ")", 
        ", lambda_mu_hat=(", paste(round(lambda_mu_dd, 5), collapse = ", "), ")"
      )
    )
  )
  
  # 予測分布のパラメータを更新:式(3.109),(3.110)
  lambda_star_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))
  mu_star_d <- m_d
  
  # 予測分布を計算:式(2.72)
  tmp_predict_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = mvnfast::dmvn(
      X = x_point_mat, mu = mu_star_d, sigma = solve(lambda_star_dd)
    ), 
    label = as.factor(
      paste0(
        "N=", n, 
        ", mu_star_hat=(", paste(round(mu_star_d, 1), collapse = ", "), ")", 
        ", lambda_star_hat=(", paste(round(lambda_star_dd, 5), collapse = ", "), ")"
      )
    )
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
  
  # 動作確認
  print(paste0("n=", n, " (", round(n / N * 100, 1), "%)"))
}

# 観測データを確認
summary(x_nd)


### 作図処理 -----

# 作図用に真のmuのデータフレームを作成
mu_df <- tibble(
  mu_1 = mu_truth_d[1], 
  mu_2 = mu_truth_d[2]
)

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_contour(data = posterior_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # muの事後分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu[1]), y = expression(mu[2]), 
       color = "density")

# gif画像を作成
gganimate::animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度を計算:式(2.72)
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_truth_d, sigma = solve(lambda_dd)
  )
)

# 作図用に観測データのデータフレームを作成
label_vec <- unique(predict_df[["label"]]) # ラベルを抽出
x_df <- tibble(x_n1 = NA, x_n2 = NA, label = label_vec[1]) # 初期値用
for(n in 1:N) {
  # n個目までのデータフレームを作成
  tmp_x_df <- tibble(
    x_n1 = x_nd[1:n, 1], 
    x_n2 = x_nd[1:n, 2], 
    label = label_vec[n + 1]
  )
  
  # 結合
  x_df <- rbind(x_df, tmp_x_df)
}

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_contour(data = predict_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 予測分布
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")

# gif画像を作成
gganimate::animate(predict_graph, nframes = N + 1, fps = 10)


